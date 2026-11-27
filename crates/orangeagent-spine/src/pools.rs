use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::{Origin, SpineError};

/// Concurrency limits for the foreground user pool vs background system pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolConfig {
    pub user_concurrency: usize,
    pub system_concurrency: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            user_concurrency: 4,
            system_concurrency: 4,
        }
    }
}

/// Point-in-time slot utilization snapshot across both pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    pub user_capacity: usize,
    pub user_available: usize,
    pub user_in_flight: usize,
    pub system_capacity: usize,
    pub system_available: usize,
    pub system_in_flight: usize,
}

/// Physically isolated concurrency pools (`UserPool` vs `SystemPool`).
///
/// Invariant: [`Origin::User`] and [`Origin::Cron`]/[`Origin::Subagent`] acquire
/// permits from completely separate [`Semaphore`]s and **never borrow across pools**.
/// Saturating the system pool with background jobs cannot starve foreground user turns.
#[derive(Debug, Clone)]
pub struct OriginPools {
    user_sem: Arc<Semaphore>,
    user_capacity: usize,
    system_sem: Arc<Semaphore>,
    system_capacity: usize,
}

impl OriginPools {
    pub fn new(config: PoolConfig) -> Self {
        let user_capacity = config.user_concurrency.max(1);
        let system_capacity = config.system_concurrency.max(1);
        Self {
            user_sem: Arc::new(Semaphore::new(user_capacity)),
            user_capacity,
            system_sem: Arc::new(Semaphore::new(system_capacity)),
            system_capacity,
        }
    }

    /// Acquire an execution slot for `origin`, aborting cleanly if `cancel` fires.
    pub async fn acquire(
        &self,
        origin: Origin,
        cancel: &CancellationToken,
    ) -> Result<OwnedSemaphorePermit, SpineError> {
        if cancel.is_cancelled() {
            return Err(SpineError::Cancelled);
        }
        let sem = self.semaphore_for(origin).clone();
        tokio::select! {
            biased;
            () = cancel.cancelled() => Err(SpineError::Cancelled),
            res = sem.acquire_owned() => res.map_err(|_| SpineError::LaneClosed),
        }
    }

    /// Non-blocking attempt to acquire a slot for `origin`.
    pub fn try_acquire(&self, origin: Origin) -> Option<OwnedSemaphorePermit> {
        self.semaphore_for(origin).clone().try_acquire_owned().ok()
    }

    /// Inspect capacity and in-flight counts for both isolated pools.
    pub fn stats(&self) -> PoolStats {
        let user_available = self.user_sem.available_permits();
        let system_available = self.system_sem.available_permits();
        PoolStats {
            user_capacity: self.user_capacity,
            user_available,
            user_in_flight: self.user_capacity.saturating_sub(user_available),
            system_capacity: self.system_capacity,
            system_available,
            system_in_flight: self.system_capacity.saturating_sub(system_available),
        }
    }

    fn semaphore_for(&self, origin: Origin) -> &Arc<Semaphore> {
        match origin {
            Origin::User => &self.user_sem,
            Origin::Cron | Origin::Subagent => &self.system_sem,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn origin_pools_isolate_user_and_system_slots_without_borrowing() {
        let pools = OriginPools::new(PoolConfig {
            user_concurrency: 1,
            system_concurrency: 2,
        });

        let sys_p1 = pools
            .try_acquire(Origin::Cron)
            .expect("first system slot available");
        let sys_p2 = pools
            .try_acquire(Origin::Subagent)
            .expect("second system slot available");
        assert!(
            pools.try_acquire(Origin::Cron).is_none(),
            "system pool should be saturated without borrowing from user pool"
        );

        let user_p = pools
            .acquire(Origin::User, &CancellationToken::new())
            .await
            .expect("user pool remains available while system pool is full");

        let stats = pools.stats();
        assert_eq!(stats.user_capacity, 1);
        assert_eq!(stats.user_available, 0);
        assert_eq!(stats.user_in_flight, 1);
        assert_eq!(stats.system_capacity, 2);
        assert_eq!(stats.system_available, 0);
        assert_eq!(stats.system_in_flight, 2);

        drop((sys_p1, sys_p2, user_p));
        let after = pools.stats();
        assert_eq!(after.user_in_flight, 0);
        assert_eq!(after.system_in_flight, 0);
    }
}
