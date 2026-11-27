use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use orangeagent_core::{SteerQueue, TurnOutcome};
use tokio::sync::{Notify, oneshot};
use tokio_util::sync::CancellationToken;

use crate::{Origin, OriginPools, SpineError, TurnContext, TurnExecutor, TurnReceipt};

/// RAII-guarded `oneshot::Sender` that decrements `in_flight_work` exactly once
/// whether resolved normally or dropped on cancellation/panic.
#[derive(Debug)]
pub(crate) struct TrackedWaiter {
    tx: Option<oneshot::Sender<Result<TurnReceipt, SpineError>>>,
    in_flight_work: Arc<AtomicUsize>,
    drain_notify: Arc<Notify>,
}

impl TrackedWaiter {
    pub(crate) fn new(
        tx: oneshot::Sender<Result<TurnReceipt, SpineError>>,
        in_flight_work: Arc<AtomicUsize>,
        drain_notify: Arc<Notify>,
    ) -> Self {
        in_flight_work.fetch_add(1, Ordering::SeqCst);
        Self {
            tx: Some(tx),
            in_flight_work,
            drain_notify,
        }
    }

    pub(crate) fn resolve(mut self, result: Result<TurnReceipt, SpineError>) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(result);
        }
    }
}

impl Drop for TrackedWaiter {
    fn drop(&mut self) {
        if self.in_flight_work.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.drain_notify.notify_waiters();
        }
    }
}

#[derive(Debug)]
pub(crate) struct InjectEntry {
    origin: Origin,
    text: String,
    waiter: TrackedWaiter,
}

#[derive(Debug)]
pub(crate) struct QueuedTurn {
    origin: Origin,
    text: String,
    is_interrupt: bool,
    waiters: Vec<TrackedWaiter>,
}

#[derive(Debug)]
struct ActiveTurn {
    turn_id: String,
    cancel: CancellationToken,
    steer: SteerQueue,
    injects: Vec<InjectEntry>,
}

#[derive(Debug)]
struct LaneState {
    active: Option<ActiveTurn>,
    pending: VecDeque<QueuedTurn>,
    last_active: Instant,
    worker_running: bool,
}

pub(crate) enum EnqueueOutcome {
    NeedsWorkerSpawn,
    WorkerAlreadyRunning,
}

/// Per-session serialized execution lane with an in-turn `Inject` mailbox and
/// idle TTL retirement.
#[derive(Debug)]
pub(crate) struct SessionLane {
    session_id: String,
    state: Mutex<LaneState>,
    notify: Notify,
}

impl SessionLane {
    pub(crate) fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            state: Mutex::new(LaneState {
                active: None,
                pending: VecDeque::new(),
                last_active: Instant::now(),
                worker_running: false,
            }),
            notify: Notify::new(),
        }
    }

    pub(crate) fn enqueue(
        &self,
        origin: Origin,
        text: String,
        policy: crate::BusyPolicy,
        waiter: TrackedWaiter,
    ) -> EnqueueOutcome {
        let mut guard = self.lock_state();
        guard.last_active = Instant::now();

        match policy {
            crate::BusyPolicy::Append => {
                guard.pending.push_back(QueuedTurn {
                    origin,
                    text,
                    is_interrupt: false,
                    waiters: vec![waiter],
                });
            }
            crate::BusyPolicy::Inject => {
                if let Some(active) = guard.active.as_mut() {
                    active.steer.push(text.clone());
                    active.injects.push(InjectEntry {
                        origin,
                        text,
                        waiter,
                    });
                } else {
                    guard.pending.push_back(QueuedTurn {
                        origin,
                        text,
                        is_interrupt: false,
                        waiters: vec![waiter],
                    });
                }
            }
            crate::BusyPolicy::Interrupt => {
                if let Some(active) = guard.active.as_ref() {
                    active.cancel.cancel();
                }
                guard.pending.push_front(QueuedTurn {
                    origin,
                    text,
                    is_interrupt: true,
                    waiters: vec![waiter],
                });
            }
        }

        let outcome = if guard.worker_running {
            EnqueueOutcome::WorkerAlreadyRunning
        } else {
            guard.worker_running = true;
            EnqueueOutcome::NeedsWorkerSpawn
        };
        drop(guard);
        self.notify.notify_one();
        outcome
    }

    pub(crate) fn notify_worker(&self) {
        self.notify.notify_one();
    }

    pub(crate) fn cancel_pending_on_cascade(&self) {
        let drained: Vec<QueuedTurn> = {
            let mut guard = self.lock_state();
            guard.pending.drain(..).collect()
        };
        for item in drained {
            resolve_all(item.waiters, || Err(SpineError::Cancelled));
        }
        self.notify.notify_one();
    }

    pub(crate) fn is_reapable(&self, idle_ttl: Duration) -> bool {
        let guard = self.lock_state();
        guard.active.is_none()
            && guard.pending.is_empty()
            && !guard.worker_running
            && guard.last_active.elapsed() >= idle_ttl
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, LaneState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

pub(crate) struct WorkerShared {
    pub(crate) executor: Arc<dyn TurnExecutor>,
    pub(crate) pools: OriginPools,
    pub(crate) sealed: Arc<AtomicBool>,
    pub(crate) cascade_token: CancellationToken,
    pub(crate) next_turn_seq: Arc<AtomicU64>,
    pub(crate) idle_ttl: Duration,
}

pub(crate) fn spawn_lane_worker(lane: Arc<SessionLane>, shared: Arc<WorkerShared>) {
    tokio::spawn(async move {
        loop {
            let maybe_next = {
                let mut guard = lane.lock_state();
                if let Some(next) = guard.pending.pop_front() {
                    Some(next)
                } else if shared.sealed.load(Ordering::SeqCst) {
                    guard.worker_running = false;
                    None
                } else {
                    None
                }
            };

            let Some(queued) = maybe_next else {
                if shared.sealed.load(Ordering::SeqCst) {
                    break;
                }
                tokio::select! {
                    () = lane.notify.notified() => continue,
                    () = tokio::time::sleep(shared.idle_ttl) => {
                        let mut guard = lane.lock_state();
                        if guard.pending.is_empty() {
                            guard.worker_running = false;
                            break;
                        }
                        continue;
                    }
                }
            };

            if shared.cascade_token.is_cancelled() {
                resolve_all(queued.waiters, || Err(SpineError::Cancelled));
                continue;
            }

            run_single_turn(&lane, &shared, queued).await;
        }
    });
}

async fn run_single_turn(lane: &SessionLane, shared: &WorkerShared, queued: QueuedTurn) {
    let seq = shared.next_turn_seq.fetch_add(1, Ordering::Relaxed);
    let turn_id = format!("{}-t{seq}", lane.session_id);
    let turn_cancel = shared.cascade_token.child_token();
    let steer = SteerQueue::new();

    {
        let mut guard = lane.lock_state();
        guard.active = Some(ActiveTurn {
            turn_id: turn_id.clone(),
            cancel: turn_cancel.clone(),
            steer: steer.clone(),
            injects: Vec::new(),
        });
    }

    let Ok(permit) = shared.pools.acquire(queued.origin, &turn_cancel).await else {
        let mut guard = lane.lock_state();
        let active = guard.active.take();
        guard.last_active = Instant::now();
        drop(guard);

        if let Some(active) = active {
            resolve_injects_on_abort(active.injects, shared.cascade_token.is_cancelled(), lane);
        }

        if shared.cascade_token.is_cancelled() {
            resolve_all(queued.waiters, || Err(SpineError::Cancelled));
        } else {
            resolve_all(queued.waiters, || {
                Ok(TurnReceipt::Executed {
                    turn_id: turn_id.clone(),
                    outcome: TurnOutcome::Interrupted {
                        reason: "interrupted while waiting for pool slot".to_string(),
                        iterations: 0,
                        total_usage: orangeagent_ai::TokenUsage::default(),
                    },
                })
            });
        }
        return;
    };

    let ctx = TurnContext {
        turn_id,
        session_id: lane.session_id.clone(),
        origin: queued.origin,
        text: queued.text,
        steer,
        cancel: turn_cancel,
    };
    let outcome = shared.executor.execute(ctx).await;
    drop(permit);

    finalize_turn(lane, shared, &outcome, queued.waiters);
}

fn finalize_turn(
    lane: &SessionLane,
    shared: &WorkerShared,
    outcome: &TurnOutcome,
    waiters: Vec<TrackedWaiter>,
) {
    let (turn_id, consumed, unconsumed) = {
        let mut guard = lane.lock_state();
        guard.last_active = Instant::now();
        let Some(mut active) = guard.active.take() else {
            return;
        };

        let leftover = active.steer.drain_all();
        let consumed_count = active.injects.len().saturating_sub(leftover.len());
        let unconsumed = active.injects.split_off(consumed_count);
        let consumed = active.injects;

        if !unconsumed.is_empty() && !shared.cascade_token.is_cancelled() {
            let degraded = merge_unconsumed_injects(unconsumed);
            let insert_idx = guard
                .pending
                .iter()
                .position(|q| !q.is_interrupt)
                .unwrap_or(guard.pending.len());
            guard.pending.insert(insert_idx, degraded);
            (active.turn_id, consumed, Vec::new())
        } else {
            (active.turn_id, consumed, unconsumed)
        }
    };

    for entry in consumed {
        entry.waiter.resolve(Ok(TurnReceipt::Injected {
            target_turn_id: turn_id.clone(),
        }));
    }
    for entry in unconsumed {
        entry.waiter.resolve(Err(SpineError::Cancelled));
    }
    resolve_all(waiters, || {
        Ok(TurnReceipt::Executed {
            turn_id: turn_id.clone(),
            outcome: outcome.clone(),
        })
    });
}

fn resolve_injects_on_abort(
    injects: Vec<InjectEntry>,
    cascade_cancelled: bool,
    lane: &SessionLane,
) {
    if injects.is_empty() {
        return;
    }
    if cascade_cancelled {
        for entry in injects {
            entry.waiter.resolve(Err(SpineError::Cancelled));
        }
        return;
    }
    let degraded = merge_unconsumed_injects(injects);
    let mut guard = lane.lock_state();
    let insert_idx = guard
        .pending
        .iter()
        .position(|q| !q.is_interrupt)
        .unwrap_or(guard.pending.len());
    guard.pending.insert(insert_idx, degraded);
}

fn merge_unconsumed_injects(entries: Vec<InjectEntry>) -> QueuedTurn {
    let origin = entries.first().map_or(Origin::User, |e| e.origin);
    let mut texts = Vec::with_capacity(entries.len());
    let mut waiters = Vec::with_capacity(entries.len());
    for entry in entries {
        texts.push(entry.text);
        waiters.push(entry.waiter);
    }
    QueuedTurn {
        origin,
        text: texts.join("\n"),
        is_interrupt: false,
        waiters,
    }
}

fn resolve_all(
    waiters: Vec<TrackedWaiter>,
    mut make: impl FnMut() -> Result<TurnReceipt, SpineError>,
) {
    for waiter in waiters {
        waiter.resolve(make());
    }
}
