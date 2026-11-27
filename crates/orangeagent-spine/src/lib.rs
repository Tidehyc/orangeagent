//! `orangeagent-spine` — Multi-session scheduling kernel (`OriginPools`, per-session
//! `SessionLane`, `BusyPolicy`, 4-phase graceful shutdown), cross-process `flock` locks,
//! tree-structured append-only JSONL session persistence, and rebuildable SQLite WAL
//! derived index with on-demand `memory_search`.

mod lane;
pub mod lock;
pub mod memory_search;
pub mod pools;
pub mod scheduler;
pub mod session;
pub mod turn;

pub use lock::{LockLease, ProcessFileLock};
pub use memory_search::MemorySearchTool;
pub use pools::{OriginPools, PoolConfig, PoolStats};
pub use scheduler::{SchedulerConfig, ShutdownReport, SpineScheduler};
pub use session::jsonl::{
    EntryKind, JsonlSessionStore, LoadReport, SessionEntry, SessionStoreError,
};
pub use session::sqlite::{
    MemorySearchHit, RebuildReport, SessionRecord, SqliteIndexError, SqliteSessionIndex,
    UsageLedgerEntry, UsageSummary,
};
pub use turn::{
    BusyPolicy, Origin, SpineError, TurnContext, TurnExecutor, TurnHandle, TurnReceipt, TurnRequest,
};
