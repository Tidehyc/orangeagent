use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{Notify, oneshot};
use tokio_util::sync::CancellationToken;

use crate::lane::{EnqueueOutcome, SessionLane, TrackedWaiter, WorkerShared, spawn_lane_worker};
use crate::{
    OriginPools, PoolConfig, PoolStats, SpineError, TurnExecutor, TurnHandle, TurnRequest,
};

/// Configuration governing [`SpineScheduler`] concurrency pools, lane reaping,
/// and 4-phase graceful shutdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub pools: PoolConfig,
    pub lane_idle_ttl: Duration,
    pub shutdown_grace: Duration,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            pools: PoolConfig::default(),
            lane_idle_ttl: Duration::from_secs(300),
            shutdown_grace: Duration::from_secs(5),
        }
    }
}

/// Summary returned by [`SpineScheduler::shutdown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShutdownReport {
    pub reached_cascade: bool,
    pub remaining_at_seal: usize,
    pub elapsed: Duration,
}

/// Multi-session concurrency scheduler enforcing:
///
/// 1. Physical isolation between foreground user turns and background jobs ([`OriginPools`])
/// 2. Per-session serialization (`SessionLane`) with `Append` / `Inject` / `Interrupt` semantics
/// 3. 4-phase graceful shutdown (`Seal -> Drain -> Grace -> Cascade`)
#[derive(Clone)]
pub struct SpineScheduler {
    lanes: Arc<Mutex<HashMap<String, Arc<SessionLane>>>>,
    shared: Arc<WorkerShared>,
    in_flight_work: Arc<AtomicUsize>,
    drain_notify: Arc<Notify>,
    shutdown_grace: Duration,
}

impl SpineScheduler {
    pub fn new(executor: Arc<dyn TurnExecutor>, config: SchedulerConfig) -> Self {
        Self {
            lanes: Arc::new(Mutex::new(HashMap::new())),
            shared: Arc::new(WorkerShared {
                executor,
                pools: OriginPools::new(config.pools),
                sealed: Arc::new(AtomicBool::new(false)),
                cascade_token: CancellationToken::new(),
                next_turn_seq: Arc::new(AtomicU64::new(1)),
                idle_ttl: config.lane_idle_ttl,
            }),
            in_flight_work: Arc::new(AtomicUsize::new(0)),
            drain_notify: Arc::new(Notify::new()),
            shutdown_grace: config.shutdown_grace,
        }
    }

    /// Submit a turn request into its session lane, returning an awaitable [`TurnHandle`].
    ///
    /// Returns [`SpineError::Sealed`] immediately once [`Self::shutdown`] has entered `Seal`.
    pub fn submit(&self, req: TurnRequest) -> Result<TurnHandle, SpineError> {
        let (tx, rx) = oneshot::channel();
        let waiter = TrackedWaiter::new(
            tx,
            Arc::clone(&self.in_flight_work),
            Arc::clone(&self.drain_notify),
        );

        if self.shared.sealed.load(Ordering::SeqCst) {
            drop(waiter);
            return Err(SpineError::Sealed);
        }

        let lane = self.get_or_create_lane(&req.session_id);
        let session_id = req.session_id;
        if let EnqueueOutcome::NeedsWorkerSpawn =
            lane.enqueue(req.origin, req.text, req.busy_policy, waiter)
        {
            spawn_lane_worker(Arc::clone(&lane), Arc::clone(&self.shared));
        }

        Ok(TurnHandle::new(session_id, rx))
    }

    /// Execute the 4-phase graceful shutdown (`Seal -> Drain -> Grace -> Cascade`).
    pub async fn shutdown(&self, grace_override: Option<Duration>) -> ShutdownReport {
        let start = Instant::now();
        // Phase 1: Seal — stop accepting new requests and wake idle lane workers to exit.
        self.shared.sealed.store(true, Ordering::SeqCst);
        let remaining_at_seal = self.in_flight_work.load(Ordering::SeqCst);
        let lanes_snapshot = self.snapshot_lanes();
        for lane in &lanes_snapshot {
            lane.notify_worker();
        }

        // Phase 2 & 3: Drain within Grace window.
        let grace = grace_override.unwrap_or(self.shutdown_grace);
        let deadline = tokio::time::Instant::now() + grace;
        let drained = self.wait_for_drain_until(Some(deadline)).await;

        let reached_cascade = !drained;
        if reached_cascade {
            // Phase 4: Cascade — cancel active turns & drain pending queues, then await full exit.
            self.shared.cascade_token.cancel();
            for lane in &lanes_snapshot {
                lane.cancel_pending_on_cascade();
            }
            let _ = self.wait_for_drain_until(None).await;
        }

        ShutdownReport {
            reached_cascade,
            remaining_at_seal,
            elapsed: start.elapsed(),
        }
    }

    /// Reap idle session lanes whose TTL has elapsed.
    pub fn reap_idle_lanes(&self) -> usize {
        let mut guard = self.lock_lanes();
        let before = guard.len();
        guard.retain(|_, lane| !lane.is_reapable(self.shared.idle_ttl));
        before.saturating_sub(guard.len())
    }

    pub fn active_lane_count(&self) -> usize {
        self.lock_lanes().len()
    }

    pub fn pool_stats(&self) -> PoolStats {
        self.shared.pools.stats()
    }

    pub fn is_sealed(&self) -> bool {
        self.shared.sealed.load(Ordering::SeqCst)
    }

    fn get_or_create_lane(&self, session_id: &str) -> Arc<SessionLane> {
        let mut guard = self.lock_lanes();
        guard
            .entry(session_id.to_string())
            .or_insert_with(|| Arc::new(SessionLane::new(session_id)))
            .clone()
    }

    fn snapshot_lanes(&self) -> Vec<Arc<SessionLane>> {
        self.lock_lanes().values().cloned().collect()
    }

    fn lock_lanes(&self) -> std::sync::MutexGuard<'_, HashMap<String, Arc<SessionLane>>> {
        self.lanes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    async fn wait_for_drain_until(&self, deadline: Option<tokio::time::Instant>) -> bool {
        loop {
            let notified = self.drain_notify.notified();
            if self.in_flight_work.load(Ordering::SeqCst) == 0 {
                return true;
            }
            match deadline {
                Some(dl) => {
                    if tokio::time::timeout_at(dl, notified).await.is_err() {
                        return self.in_flight_work.load(Ordering::SeqCst) == 0;
                    }
                }
                None => notified.await,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use async_trait::async_trait;
    use orangeagent_ai::TokenUsage;
    use orangeagent_core::TurnOutcome;
    use tokio::sync::{Barrier, Mutex as AsyncMutex};

    use super::*;
    use crate::{BusyPolicy, TurnContext, TurnReceipt};

    struct ScriptedExecutor {
        events: AsyncMutex<Vec<String>>,
        delay: Duration,
        in_flight_user: AtomicUsize,
        in_flight_sys: AtomicUsize,
        consume_steer_immediately: bool,
        block_on_gate: Option<Arc<Notify>>,
    }

    impl ScriptedExecutor {
        fn new(delay: Duration) -> Self {
            Self {
                events: AsyncMutex::new(Vec::new()),
                delay,
                in_flight_user: AtomicUsize::new(0),
                in_flight_sys: AtomicUsize::new(0),
                consume_steer_immediately: false,
                block_on_gate: None,
            }
        }
    }

    #[async_trait]
    impl TurnExecutor for ScriptedExecutor {
        async fn execute(&self, ctx: TurnContext) -> TurnOutcome {
            if ctx.origin.is_user() {
                self.in_flight_user.fetch_add(1, Ordering::SeqCst);
            } else {
                self.in_flight_sys.fetch_add(1, Ordering::SeqCst);
            }

            self.events
                .lock()
                .await
                .push(format!("start:{}:{}", ctx.session_id, ctx.text));

            if let Some(gate) = &self.block_on_gate
                && ctx.text.contains("gate")
            {
                tokio::select! {
                    () = ctx.cancel.cancelled() => {
                        return finish_interrupted(ctx.origin, &self.in_flight_user, &self.in_flight_sys);
                    }
                    () = gate.notified() => {}
                }
            }

            if !self.delay.is_zero() {
                tokio::select! {
                    () = ctx.cancel.cancelled() => {
                        return finish_interrupted(ctx.origin, &self.in_flight_user, &self.in_flight_sys);
                    }
                    () = tokio::time::sleep(self.delay) => {}
                }
            }

            let mut final_text = ctx.text.clone();
            if self.consume_steer_immediately {
                let steered = ctx.steer.drain_all();
                if !steered.is_empty() {
                    final_text = format!("{final_text}+[{}]", steered.join("|"));
                }
            }

            self.events
                .lock()
                .await
                .push(format!("end:{}:{}", ctx.session_id, final_text));

            if ctx.origin.is_user() {
                self.in_flight_user.fetch_sub(1, Ordering::SeqCst);
            } else {
                self.in_flight_sys.fetch_sub(1, Ordering::SeqCst);
            }

            TurnOutcome::Completed {
                final_text,
                iterations: 1,
                total_usage: TokenUsage::default(),
            }
        }
    }

    fn finish_interrupted(
        origin: crate::Origin,
        in_flight_user: &AtomicUsize,
        in_flight_sys: &AtomicUsize,
    ) -> TurnOutcome {
        if origin.is_user() {
            in_flight_user.fetch_sub(1, Ordering::SeqCst);
        } else {
            in_flight_sys.fetch_sub(1, Ordering::SeqCst);
        }
        TurnOutcome::Interrupted {
            reason: "cancelled".into(),
            iterations: 0,
            total_usage: TokenUsage::default(),
        }
    }

    #[tokio::test]
    async fn origin_pools_prevent_background_starvation_of_foreground_user() {
        let gate = Arc::new(Notify::new());
        let exec = Arc::new(ScriptedExecutor {
            block_on_gate: Some(Arc::clone(&gate)),
            ..ScriptedExecutor::new(Duration::ZERO)
        });
        let scheduler = SpineScheduler::new(
            exec,
            SchedulerConfig {
                pools: PoolConfig {
                    user_concurrency: 1,
                    system_concurrency: 1,
                },
                lane_idle_ttl: Duration::from_secs(60),
                shutdown_grace: Duration::from_secs(2),
            },
        );

        let bg_handle = scheduler
            .submit(TurnRequest::cron("bg-cron-1", "gate:slow-bg"))
            .expect("submit bg");
        let bg_queued = scheduler
            .submit(TurnRequest::subagent("bg-sub-2", "queued-bg"))
            .expect("submit bg2");

        // Wait briefly until bg_handle holds the sole system pool permit
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert_eq!(scheduler.pool_stats().system_in_flight, 1);

        // Foreground user turn on a separate session completes immediately without waiting on gate!
        let user_handle = scheduler
            .submit(TurnRequest::user("user-main", "fast-user-query"))
            .expect("submit user");
        let user_receipt = tokio::time::timeout(Duration::from_millis(200), user_handle.wait())
            .await
            .expect("user turn must not starve behind saturated system pool")
            .expect("user turn ok");
        assert!(matches!(
            user_receipt,
            TurnReceipt::Executed {
                outcome: TurnOutcome::Completed { ref final_text, .. },
                ..
            } if final_text == "fast-user-query"
        ));

        gate.notify_waiters();
        let _ = bg_handle.wait().await.expect("bg1 ok");
        let _ = bg_queued.wait().await.expect("bg2 ok");
        let _ = scheduler.shutdown(None).await;
    }

    #[tokio::test]
    async fn session_lane_serializes_same_session_and_parallelizes_distinct_sessions() {
        struct ParallelProbeExecutor {
            barrier: Barrier,
        }

        #[async_trait]
        impl TurnExecutor for ParallelProbeExecutor {
            async fn execute(&self, ctx: TurnContext) -> TurnOutcome {
                if ctx.text == "sync" {
                    self.barrier.wait().await;
                }
                TurnOutcome::Completed {
                    final_text: ctx.text,
                    iterations: 1,
                    total_usage: TokenUsage::default(),
                }
            }
        }

        // First verify two distinct sessions actually run concurrently via a 2-party Barrier
        let barrier_sched = SpineScheduler::new(
            Arc::new(ParallelProbeExecutor {
                barrier: Barrier::new(2),
            }),
            SchedulerConfig::default(),
        );
        let h_a = barrier_sched
            .submit(TurnRequest::user("sess-a", "sync"))
            .expect("a");
        let h_b = barrier_sched
            .submit(TurnRequest::user("sess-b", "sync"))
            .expect("b");
        let (r_a, r_b) = tokio::time::timeout(Duration::from_millis(500), async {
            tokio::join!(h_a.wait(), h_b.wait())
        })
        .await
        .expect("distinct sessions must execute in parallel without deadlocking the barrier");
        assert!(r_a.is_ok() && r_b.is_ok());
        let _ = barrier_sched.shutdown(None).await;

        // Next verify same-session turns execute strictly serially in FIFO order
        let exec = Arc::new(ScriptedExecutor::new(Duration::from_millis(20)));
        let serial_sched = SpineScheduler::new(
            Arc::clone(&exec) as Arc<dyn TurnExecutor>,
            SchedulerConfig::default(),
        );
        let h1 = serial_sched
            .submit(TurnRequest::user("same", "first"))
            .expect("1");
        let h2 = serial_sched
            .submit(TurnRequest::user("same", "second"))
            .expect("2");
        let _ = tokio::join!(h1.wait(), h2.wait());

        let log = exec.events.lock().await.clone();
        assert_eq!(
            log,
            vec![
                "start:same:first",
                "end:same:first",
                "start:same:second",
                "end:same:second",
            ]
        );
        let _ = serial_sched.shutdown(None).await;
    }

    #[tokio::test]
    async fn busy_policy_inject_merges_into_active_turn_and_degrades_when_unconsumed() {
        // Case 1: Active turn drains SteerQueue -> Inject returns TurnReceipt::Injected
        let consuming_exec = Arc::new(ScriptedExecutor {
            consume_steer_immediately: true,
            ..ScriptedExecutor::new(Duration::from_millis(40))
        });
        let sched1 = SpineScheduler::new(
            Arc::clone(&consuming_exec) as Arc<dyn TurnExecutor>,
            SchedulerConfig::default(),
        );

        let t1 = sched1
            .submit(TurnRequest::user("s1", "base"))
            .expect("submit base");
        tokio::time::sleep(Duration::from_millis(10)).await;
        let inj1 = sched1
            .submit(TurnRequest::user("s1", "extra-1").with_busy_policy(BusyPolicy::Inject))
            .expect("inject 1");

        let (r_t1, r_inj1) = tokio::join!(t1.wait(), inj1.wait());
        let TurnReceipt::Executed {
            turn_id: t1_id,
            outcome: TurnOutcome::Completed { final_text, .. },
        } = r_t1.expect("t1")
        else {
            panic!("expected executed outcome");
        };
        assert_eq!(final_text, "base+[extra-1]");
        assert_eq!(
            r_inj1.expect("inj1"),
            TurnReceipt::Injected {
                target_turn_id: t1_id
            }
        );
        let _ = sched1.shutdown(None).await;

        // Case 2: Active turn finishes without draining SteerQueue -> two Injects merge into
        // one degraded Append turn so zero messages are ever lost!
        let ignoring_exec = Arc::new(ScriptedExecutor {
            consume_steer_immediately: false,
            ..ScriptedExecutor::new(Duration::from_millis(40))
        });
        let sched2 = SpineScheduler::new(
            Arc::clone(&ignoring_exec) as Arc<dyn TurnExecutor>,
            SchedulerConfig::default(),
        );

        let t2 = sched2
            .submit(TurnRequest::user("s2", "primary"))
            .expect("submit primary");
        tokio::time::sleep(Duration::from_millis(10)).await;
        let inj_a = sched2
            .submit(TurnRequest::user("s2", "follow-a").with_busy_policy(BusyPolicy::Inject))
            .expect("inject a");
        let inj_b = sched2
            .submit(TurnRequest::user("s2", "follow-b").with_busy_policy(BusyPolicy::Inject))
            .expect("inject b");

        let (r_t2, r_a, r_b) = tokio::join!(t2.wait(), inj_a.wait(), inj_b.wait());
        assert!(matches!(r_t2.expect("t2"), TurnReceipt::Executed { .. }));
        let ra = r_a.expect("degraded a");
        let rb = r_b.expect("degraded b");
        assert_eq!(ra, rb);
        assert!(matches!(
            ra,
            TurnReceipt::Executed {
                outcome: TurnOutcome::Completed { ref final_text, .. },
                ..
            } if final_text == "follow-a\nfollow-b"
        ));
        let _ = sched2.shutdown(None).await;
    }

    #[tokio::test]
    async fn busy_policy_interrupt_cancels_active_turn_and_jumps_ahead_of_pending() {
        let exec = Arc::new(ScriptedExecutor::new(Duration::from_millis(250)));
        let sched = SpineScheduler::new(
            Arc::clone(&exec) as Arc<dyn TurnExecutor>,
            SchedulerConfig::default(),
        );

        let active = sched
            .submit(TurnRequest::user("sess-int", "slow-active"))
            .expect("active");
        let queued = sched
            .submit(TurnRequest::user("sess-int", "normal-queued"))
            .expect("queued");
        tokio::time::sleep(Duration::from_millis(20)).await;

        let urgent = sched
            .submit(
                TurnRequest::user("sess-int", "urgent-interrupt")
                    .with_busy_policy(BusyPolicy::Interrupt),
            )
            .expect("interrupt");

        let (r_active, r_urgent, r_queued) =
            tokio::join!(active.wait(), urgent.wait(), queued.wait());
        assert!(matches!(
            r_active.expect("active receipt"),
            TurnReceipt::Executed {
                outcome: TurnOutcome::Interrupted { .. },
                ..
            }
        ));
        assert!(matches!(
            r_urgent.expect("urgent receipt"),
            TurnReceipt::Executed {
                outcome: TurnOutcome::Completed { .. },
                ..
            }
        ));
        assert!(matches!(
            r_queued.expect("queued receipt"),
            TurnReceipt::Executed {
                outcome: TurnOutcome::Completed { .. },
                ..
            }
        ));

        let log = exec.events.lock().await.clone();
        let urgent_idx = log
            .iter()
            .position(|e| e == "start:sess-int:urgent-interrupt")
            .expect("urgent started");
        let queued_idx = log
            .iter()
            .position(|e| e == "start:sess-int:normal-queued")
            .expect("queued started");
        assert!(
            urgent_idx < queued_idx,
            "Interrupt must jump ahead of normal pending turns: {log:?}"
        );
        let _ = sched.shutdown(None).await;
    }

    #[tokio::test]
    async fn four_phase_shutdown_drains_cleanly_and_cascades_on_grace_timeout() {
        // Phase A: clean drain within grace window
        let fast_exec = Arc::new(ScriptedExecutor::new(Duration::from_millis(30)));
        let fast_sched = SpineScheduler::new(fast_exec, SchedulerConfig::default());
        let h = fast_sched
            .submit(TurnRequest::user("s-fast", "quick"))
            .expect("submit");
        let report_clean = fast_sched.shutdown(Some(Duration::from_millis(500))).await;
        assert!(!report_clean.reached_cascade);
        assert_eq!(report_clean.remaining_at_seal, 1);
        assert!(matches!(
            fast_sched.submit(TurnRequest::user("s-fast", "after-seal")),
            Err(SpineError::Sealed)
        ));
        assert!(h.wait().await.is_ok());

        // Phase B: stuck active turn + queued turn -> grace expires -> cascade cancels both
        let gate = Arc::new(Notify::new());
        let stuck_exec = Arc::new(ScriptedExecutor {
            block_on_gate: Some(gate),
            ..ScriptedExecutor::new(Duration::ZERO)
        });
        let stuck_sched = SpineScheduler::new(stuck_exec, SchedulerConfig::default());
        let h_active = stuck_sched
            .submit(TurnRequest::user("s-stuck", "gate:hang"))
            .expect("stuck active");
        let h_pending = stuck_sched
            .submit(TurnRequest::user("s-stuck", "waiting-in-queue"))
            .expect("stuck pending");
        tokio::time::sleep(Duration::from_millis(20)).await;

        let report_cascade = stuck_sched.shutdown(Some(Duration::from_millis(40))).await;
        assert!(report_cascade.reached_cascade);
        assert_eq!(report_cascade.remaining_at_seal, 2);

        assert!(matches!(
            h_active.wait().await.expect("active finishes interrupted"),
            TurnReceipt::Executed {
                outcome: TurnOutcome::Interrupted { .. },
                ..
            }
        ));
        assert!(matches!(h_pending.wait().await, Err(SpineError::Cancelled)));
    }

    #[tokio::test]
    async fn idle_lane_reaper_cleans_up_expired_lanes() {
        let exec = Arc::new(ScriptedExecutor::new(Duration::ZERO));
        let sched = SpineScheduler::new(
            exec,
            SchedulerConfig {
                pools: PoolConfig::default(),
                lane_idle_ttl: Duration::from_millis(20),
                shutdown_grace: Duration::from_secs(1),
            },
        );

        let h = sched
            .submit(TurnRequest::user("ephemeral-lane", "ping"))
            .expect("submit");
        let _ = h.wait().await.expect("ok");
        assert_eq!(sched.active_lane_count(), 1);

        tokio::time::sleep(Duration::from_millis(60)).await;
        let reaped = sched.reap_idle_lanes();
        assert_eq!(reaped, 1);
        assert_eq!(sched.active_lane_count(), 0);
        let _ = sched.shutdown(None).await;
    }
}
