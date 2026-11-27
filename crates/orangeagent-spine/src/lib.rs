//! `orangeagent-spine` — Multi-session scheduling kernel (`OriginPools`, per-session
//! `SessionLane`, `BusyPolicy`, 4-phase graceful shutdown), cross-process `flock` locks,
//! and tree-structured append-only JSONL session persistence.

mod lane;
pub mod lock;
pub mod pools;
pub mod scheduler;
pub mod session;
pub mod turn;

pub use lock::{LockLease, ProcessFileLock};
pub use pools::{OriginPools, PoolConfig, PoolStats};
pub use scheduler::{SchedulerConfig, ShutdownReport, SpineScheduler};
pub use session::jsonl::{
    EntryKind, JsonlSessionStore, LoadReport, SessionEntry, SessionStoreError,
};
pub use turn::{
    BusyPolicy, Origin, SpineError, TurnContext, TurnExecutor, TurnHandle, TurnReceipt, TurnRequest,
};
