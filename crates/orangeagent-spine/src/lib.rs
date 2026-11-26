//! `orangeagent-spine` — Multi-session scheduling kernel, tree-structured append-only
//! JSONL session persistence, and (later in M2/M4) SQLite WAL derived index, OriginPools,
//! Lane serialization, and Daemon/ACP transports.

pub mod session;

pub use session::jsonl::{
    EntryKind, JsonlSessionStore, LoadReport, SessionEntry, SessionStoreError,
};
