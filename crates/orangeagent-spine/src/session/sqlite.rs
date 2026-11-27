use std::path::{Path, PathBuf};
use std::sync::Mutex;

use orangeagent_ai::{ContentBlock, PricingCatalog, Role};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::session::jsonl::{EntryKind, JsonlSessionStore, SessionStoreError};

const MAX_TOOL_RESULT_FTS_CHARS: usize = 4_000;

/// Errors raised by [`SqliteSessionIndex`].
#[derive(Debug, Error)]
pub enum SqliteIndexError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("sqlite index I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("session store error during index rebuild: {0}")]
    SessionStore(#[from] SessionStoreError),

    #[error("invalid memory search query: {0}")]
    InvalidQuery(String),
}

/// Row stored in the derived `sessions` metadata table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionRecord {
    pub session_id: String,
    pub workspace_hash: String,
    pub title: Option<String>,
    pub active_leaf_id: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub total_cost_usd: f64,
}

/// Row stored in the `usage_ledger` cost and cache-hit table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageLedgerEntry {
    pub session_id: String,
    pub turn_id: String,
    pub provider: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub cost_usd: f64,
    pub latency_ms: u64,
    pub created_at_ms: u64,
}

/// Aggregated token, cache, and USD cost totals from `usage_ledger`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageSummary {
    pub call_count: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub total_cost_usd: f64,
}

impl UsageSummary {
    /// Prompt cache hit ratio across all summed calls (`cache_read / input_tokens`).
    #[allow(clippy::cast_precision_loss)]
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.input_tokens == 0 {
            return 0.0;
        }
        (self.cache_read as f64 / self.input_tokens as f64).clamp(0.0, 1.0)
    }
}

/// Single ranked cross-session hit returned by [`SqliteSessionIndex::search_memory`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemorySearchHit {
    pub session_id: String,
    pub session_title: Option<String>,
    pub entry_id: String,
    pub role: String,
    pub snippet: String,
    /// SQLite FTS5 `bm25()` score (lower / more negative indicates a stronger match).
    pub bm25_score: f64,
}

/// Telemetry returned by [`SqliteSessionIndex::rebuild_from_jsonl_dir`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RebuildReport {
    pub sessions_indexed: usize,
    pub fts_entries_indexed: usize,
    pub usage_rows_indexed: usize,
    pub skipped_files: usize,
}

/// Rebuildable SQLite WAL derived index (`sessions`, `session_fts` FTS5, `usage_ledger`).
///
/// Architectural invariant: append-only JSONL files are the **sole Source of Truth**.
/// If `index.sqlite` is deleted or corrupted, [`Self::rebuild_from_jsonl_dir`]
/// reconstructs every table deterministically from the `.jsonl` files on disk.
#[derive(Debug)]
pub struct SqliteSessionIndex {
    path: PathBuf,
    conn: Mutex<Connection>,
}

impl SqliteSessionIndex {
    /// Open (or create) `index.sqlite` with WAL mode, `busy_timeout`, and schema DDL.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SqliteIndexError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA busy_timeout = 5000;
             PRAGMA foreign_keys = ON;",
        )?;
        init_schema(&conn)?;

        Ok(Self {
            path,
            conn: Mutex::new(conn),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Inspect the active SQLite `journal_mode` (expected `"wal"`).
    pub fn journal_mode(&self) -> Result<String, SqliteIndexError> {
        let guard = self.lock_conn();
        let mode: String = guard.query_row("PRAGMA journal_mode;", [], |row| row.get(0))?;
        Ok(mode)
    }

    /// Upsert one [`JsonlSessionStore`] into `sessions`, `session_fts`, and `usage_ledger`.
    ///
    /// Returns `(fts_entries_indexed, usage_rows_indexed)`.
    pub fn index_session(
        &self,
        store: &JsonlSessionStore,
        pricing: &PricingCatalog,
    ) -> Result<(usize, usize), SqliteIndexError> {
        let mut guard = self.lock_conn();
        let tx = guard.transaction()?;
        let counts = index_session_in_tx(&tx, store, pricing)?;
        tx.commit()?;
        Ok(counts)
    }

    /// Append a live [`UsageLedgerEntry`] and update the session's `total_cost_usd`.
    pub fn record_usage(&self, entry: &UsageLedgerEntry) -> Result<(), SqliteIndexError> {
        let mut guard = self.lock_conn();
        let tx = guard.transaction()?;
        tx.execute(
            "INSERT INTO usage_ledger (
                session_id, turn_id, provider, model,
                input_tokens, output_tokens, cache_read, cache_write,
                cost_usd, latency_ms, created_at_ms
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                entry.session_id,
                entry.turn_id,
                entry.provider,
                entry.model,
                to_i64(entry.input_tokens),
                to_i64(entry.output_tokens),
                to_i64(entry.cache_read),
                to_i64(entry.cache_write),
                entry.cost_usd,
                to_i64(entry.latency_ms),
                to_i64(entry.created_at_ms),
            ],
        )?;
        tx.execute(
            "UPDATE sessions
             SET total_cost_usd = total_cost_usd + ?1,
                 updated_at_ms = MAX(updated_at_ms, ?2)
             WHERE session_id = ?3",
            params![
                entry.cost_usd,
                to_i64(entry.created_at_ms),
                entry.session_id
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Wipe all derived tables and deterministically rebuild them from all `*.jsonl`
    /// files under `sessions_dir`.
    pub fn rebuild_from_jsonl_dir(
        &self,
        sessions_dir: impl AsRef<Path>,
        pricing: &PricingCatalog,
    ) -> Result<RebuildReport, SqliteIndexError> {
        let files = collect_jsonl_files(sessions_dir.as_ref())?;

        let mut guard = self.lock_conn();
        let tx = guard.transaction()?;
        tx.execute("DELETE FROM session_fts", [])?;
        tx.execute("DELETE FROM usage_ledger", [])?;
        tx.execute("DELETE FROM sessions", [])?;

        let mut report = RebuildReport::default();
        for file_path in files {
            let Ok(store) = JsonlSessionStore::open(&file_path) else {
                report.skipped_files += 1;
                continue;
            };
            let (fts_count, usage_count) = index_session_in_tx(&tx, &store, pricing)?;
            report.sessions_indexed += 1;
            report.fts_entries_indexed += fts_count;
            report.usage_rows_indexed += usage_count;
        }
        tx.commit()?;
        Ok(report)
    }

    /// List indexed sessions ordered by most recently updated first.
    pub fn list_sessions(&self, limit: usize) -> Result<Vec<SessionRecord>, SqliteIndexError> {
        let guard = self.lock_conn();
        let capped = to_i64(limit.clamp(1, 500) as u64);
        let mut stmt = guard.prepare(
            "SELECT session_id, workspace_hash, title, active_leaf_id,
                    created_at_ms, updated_at_ms, total_cost_usd
             FROM sessions
             ORDER BY updated_at_ms DESC, session_id ASC
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![capped], map_session_row)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Fetch one indexed session by ID.
    pub fn get_session(&self, session_id: &str) -> Result<Option<SessionRecord>, SqliteIndexError> {
        let guard = self.lock_conn();
        guard
            .query_row(
                "SELECT session_id, workspace_hash, title, active_leaf_id,
                        created_at_ms, updated_at_ms, total_cost_usd
                 FROM sessions
                 WHERE session_id = ?1",
                params![session_id],
                map_session_row,
            )
            .optional()
            .map_err(Into::into)
    }

    /// Aggregate token, cache, and cost metrics globally or for a single session.
    pub fn usage_summary(
        &self,
        session_id: Option<&str>,
    ) -> Result<UsageSummary, SqliteIndexError> {
        let guard = self.lock_conn();
        let map_summary = |row: &rusqlite::Row<'_>| -> rusqlite::Result<UsageSummary> {
            Ok(UsageSummary {
                call_count: to_u64(row.get::<_, i64>(0)?),
                input_tokens: to_u64(row.get::<_, i64>(1)?),
                output_tokens: to_u64(row.get::<_, i64>(2)?),
                cache_read: to_u64(row.get::<_, i64>(3)?),
                cache_write: to_u64(row.get::<_, i64>(4)?),
                total_cost_usd: row.get::<_, f64>(5)?,
            })
        };

        let summary = match session_id {
            Some(sid) => guard.query_row(
                "SELECT COUNT(*),
                        COALESCE(SUM(input_tokens), 0),
                        COALESCE(SUM(output_tokens), 0),
                        COALESCE(SUM(cache_read), 0),
                        COALESCE(SUM(cache_write), 0),
                        COALESCE(SUM(cost_usd), 0.0)
                 FROM usage_ledger
                 WHERE session_id = ?1",
                params![sid],
                map_summary,
            )?,
            None => guard.query_row(
                "SELECT COUNT(*),
                        COALESCE(SUM(input_tokens), 0),
                        COALESCE(SUM(output_tokens), 0),
                        COALESCE(SUM(cache_read), 0),
                        COALESCE(SUM(cache_write), 0),
                        COALESCE(SUM(cost_usd), 0.0)
                 FROM usage_ledger",
                [],
                map_summary,
            )?,
        };
        Ok(summary)
    }

    /// Execute an FTS5 BM25 full-text search across historical sessions.
    pub fn search_memory(
        &self,
        query: &str,
        session_filter: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemorySearchHit>, SqliteIndexError> {
        let fts_query = build_fts5_query(query)?;
        let capped = to_i64(limit.clamp(1, 50) as u64);
        let guard = self.lock_conn();

        let map_hit = |row: &rusqlite::Row<'_>| -> rusqlite::Result<MemorySearchHit> {
            Ok(MemorySearchHit {
                session_id: row.get(0)?,
                session_title: row.get(1)?,
                entry_id: row.get(2)?,
                role: row.get(3)?,
                snippet: row.get(4)?,
                bm25_score: row.get(5)?,
            })
        };

        if let Some(sid) = session_filter.map(str::trim).filter(|s| !s.is_empty()) {
            let mut stmt = guard.prepare(
                "SELECT f.session_id,
                        s.title,
                        f.entry_id,
                        f.role,
                        snippet(session_fts, 3, '[', ']', '...', 24),
                        bm25(session_fts) AS score
                 FROM session_fts f
                 LEFT JOIN sessions s ON s.session_id = f.session_id
                 WHERE session_fts MATCH ?1 AND f.session_id = ?2
                 ORDER BY score ASC
                 LIMIT ?3",
            )?;
            let rows = stmt.query_map(params![fts_query, sid, capped], map_hit)?;
            rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
        } else {
            let mut stmt = guard.prepare(
                "SELECT f.session_id,
                        s.title,
                        f.entry_id,
                        f.role,
                        snippet(session_fts, 3, '[', ']', '...', 24),
                        bm25(session_fts) AS score
                 FROM session_fts f
                 LEFT JOIN sessions s ON s.session_id = f.session_id
                 WHERE session_fts MATCH ?1
                 ORDER BY score ASC
                 LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![fts_query, capped], map_hit)?;
            rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
        }
    }

    fn lock_conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn init_schema(conn: &Connection) -> Result<(), SqliteIndexError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS sessions (
            session_id      TEXT PRIMARY KEY,
            workspace_hash  TEXT NOT NULL,
            title           TEXT,
            active_leaf_id  TEXT,
            created_at_ms   INTEGER NOT NULL,
            updated_at_ms   INTEGER NOT NULL,
            total_cost_usd  REAL NOT NULL DEFAULT 0.0
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_updated
            ON sessions (updated_at_ms DESC);

        CREATE VIRTUAL TABLE IF NOT EXISTS session_fts USING fts5(
            session_id UNINDEXED,
            entry_id UNINDEXED,
            role UNINDEXED,
            content,
            tokenize = 'unicode61'
        );

        CREATE TABLE IF NOT EXISTS usage_ledger (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            turn_id         TEXT NOT NULL,
            provider        TEXT NOT NULL,
            model           TEXT NOT NULL,
            input_tokens    INTEGER NOT NULL,
            output_tokens   INTEGER NOT NULL,
            cache_read      INTEGER NOT NULL,
            cache_write     INTEGER NOT NULL,
            cost_usd        REAL NOT NULL,
            latency_ms      INTEGER NOT NULL,
            created_at_ms   INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_usage_ledger_session
            ON usage_ledger (session_id, created_at_ms);",
    )?;
    Ok(())
}

fn index_session_in_tx(
    tx: &rusqlite::Transaction<'_>,
    store: &JsonlSessionStore,
    pricing: &PricingCatalog,
) -> Result<(usize, usize), SqliteIndexError> {
    let session_id = resolve_session_id(store);
    tx.execute(
        "DELETE FROM session_fts WHERE session_id = ?1",
        params![session_id],
    )?;
    tx.execute(
        "DELETE FROM usage_ledger WHERE session_id = ?1",
        params![session_id],
    )?;

    let mut fts_stmt = tx.prepare(
        "INSERT INTO session_fts (session_id, entry_id, role, content)
         VALUES (?1, ?2, ?3, ?4)",
    )?;
    let mut usage_stmt = tx.prepare(
        "INSERT INTO usage_ledger (
            session_id, turn_id, provider, model,
            input_tokens, output_tokens, cache_read, cache_write,
            cost_usd, latency_ms, created_at_ms
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
    )?;

    let mut fts_count = 0usize;
    let mut usage_count = 0usize;
    let mut total_cost = 0.0f64;
    let mut created_at_ms: Option<u64> = None;
    let mut updated_at_ms: u64 = 0;

    for entry in store.entries() {
        created_at_ms.get_or_insert(entry.timestamp_ms);
        updated_at_ms = updated_at_ms.max(entry.timestamp_ms);

        match &entry.kind {
            EntryKind::Message {
                message,
                model,
                usage,
                ..
            } => {
                let searchable = extract_searchable_message(message);
                if !searchable.is_empty() {
                    fts_stmt.execute(params![
                        session_id,
                        entry.id,
                        role_label(message.role),
                        searchable
                    ])?;
                    fts_count += 1;
                }

                if let Some(u) = usage {
                    total_cost += insert_indexed_usage(
                        &mut usage_stmt,
                        &session_id,
                        &entry.id,
                        entry.timestamp_ms,
                        model.as_deref(),
                        u,
                        pricing,
                    )?;
                    usage_count += 1;
                }
            }
            EntryKind::Compaction { summary, .. } if !summary.trim().is_empty() => {
                fts_stmt.execute(params![session_id, entry.id, "compaction", summary])?;
                fts_count += 1;
            }
            _ => {}
        }
    }

    drop(fts_stmt);
    drop(usage_stmt);

    let workspace_hash = fnv1a_64_hex(store.workspace_root().unwrap_or(""));
    let created = created_at_ms.unwrap_or(0);
    tx.execute(
        "INSERT INTO sessions (
            session_id, workspace_hash, title, active_leaf_id,
            created_at_ms, updated_at_ms, total_cost_usd
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(session_id) DO UPDATE SET
            workspace_hash = excluded.workspace_hash,
            title = excluded.title,
            active_leaf_id = excluded.active_leaf_id,
            created_at_ms = excluded.created_at_ms,
            updated_at_ms = excluded.updated_at_ms,
            total_cost_usd = excluded.total_cost_usd",
        params![
            session_id,
            workspace_hash,
            store.title(),
            store.active_leaf_id(),
            to_i64(created),
            to_i64(updated_at_ms),
            total_cost,
        ],
    )?;

    Ok((fts_count, usage_count))
}

fn insert_indexed_usage(
    usage_stmt: &mut rusqlite::Statement<'_>,
    session_id: &str,
    entry_id: &str,
    timestamp_ms: u64,
    model: Option<&str>,
    usage: &orangeagent_ai::TokenUsage,
    pricing: &PricingCatalog,
) -> Result<f64, SqliteIndexError> {
    let model_name = model.unwrap_or("unknown");
    let provider_name = pricing
        .lookup(model_name)
        .map_or("unknown", |p| p.provider_id.as_str());
    let cost_usd = pricing
        .estimate_cost(model_name, usage)
        .map_or(0.0, |c| c.total_usd);

    usage_stmt.execute(params![
        session_id,
        entry_id,
        provider_name,
        model_name,
        to_i64(usage.input_tokens),
        to_i64(usage.output_tokens),
        to_i64(usage.cache_read_tokens),
        to_i64(usage.cache_write_tokens),
        cost_usd,
        0_i64,
        to_i64(timestamp_ms),
    ])?;
    Ok(cost_usd)
}

fn map_session_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SessionRecord> {
    Ok(SessionRecord {
        session_id: row.get(0)?,
        workspace_hash: row.get(1)?,
        title: row.get(2)?,
        active_leaf_id: row.get(3)?,
        created_at_ms: to_u64(row.get::<_, i64>(4)?),
        updated_at_ms: to_u64(row.get::<_, i64>(5)?),
        total_cost_usd: row.get(6)?,
    })
}

fn resolve_session_id(store: &JsonlSessionStore) -> String {
    if let Some(id) = store.session_id() {
        return id.to_string();
    }
    store
        .path()
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn extract_searchable_message(message: &orangeagent_ai::Message) -> String {
    let mut parts = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::Text { text } if !text.trim().is_empty() => {
                parts.push(text.clone());
            }
            ContentBlock::ToolResult { content, .. } if !content.trim().is_empty() => {
                parts.push(truncate_chars(content, MAX_TOOL_RESULT_FTS_CHARS));
            }
            _ => {}
        }
    }
    parts.join("\n")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut iter = text.chars();
    let head: String = iter.by_ref().take(max_chars).collect();
    if iter.next().is_some() {
        format!("{head}...")
    } else {
        head
    }
}

fn role_label(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn collect_jsonl_files(dir: &Path) -> Result<Vec<PathBuf>, SqliteIndexError> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn build_fts5_query(raw: &str) -> Result<String, SqliteIndexError> {
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    if tokens.is_empty() {
        return Err(SqliteIndexError::InvalidQuery(
            "query must not be empty".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(tokens.len());
    for (idx, token) in tokens.iter().enumerate() {
        let is_binary_or = *token == "OR" && idx > 0 && idx + 1 < tokens.len();
        if is_binary_or {
            out.push("OR".to_string());
        } else {
            let escaped = token.replace('"', "\"\"");
            out.push(format!("\"{escaped}\""));
        }
    }
    Ok(out.join(" "))
}

fn fnv1a_64_hex(input: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn to_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn to_u64(value: i64) -> u64 {
    u64::try_from(value).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use orangeagent_ai::{Message, TokenUsage};
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn open_enables_wal_and_creates_schema() {
        let dir = TempDir::new().expect("tempdir");
        let db_path = dir.path().join("state").join("index.sqlite");
        let index = SqliteSessionIndex::open(&db_path).expect("open sqlite index");

        assert_eq!(index.path(), db_path);
        assert_eq!(
            index.journal_mode().expect("journal_mode").to_lowercase(),
            "wal"
        );
        assert_eq!(index.list_sessions(10).expect("empty list").len(), 0);
        assert_eq!(
            index.usage_summary(None).expect("empty usage").call_count,
            0
        );
    }

    #[test]
    fn delete_and_rebuild_from_jsonl_produces_identical_index_and_ledger() {
        let dir = TempDir::new().expect("tempdir");
        let sessions_dir = dir.path().join("sessions");
        let db_path = dir.path().join("index.sqlite");
        let pricing = PricingCatalog::builtin();

        let mut s1 = JsonlSessionStore::create(
            sessions_dir.join("s1.jsonl"),
            "sess-1",
            "/tmp/proj-a",
            Some("Fix Landlock sandbox".into()),
        )
        .expect("create s1");
        s1.append_message(
            Message::user("How does Landlock enforce workspace writes?"),
            None,
            None,
            None,
        )
        .expect("s1 u1");
        s1.append_message(
            Message::assistant_text("Landlock restricts PathBeneath to the workspace root."),
            Some("claude-sonnet-4-6".into()),
            Some(TokenUsage {
                input_tokens: 1_000,
                output_tokens: 200,
                cache_read_tokens: 800,
                cache_write_tokens: 0,
                reasoning_tokens: 0,
            }),
            None,
        )
        .expect("s1 a1");

        let mut s2 = JsonlSessionStore::create(
            sessions_dir.join("s2.jsonl"),
            "sess-2",
            "/tmp/proj-a",
            Some("Tune prompt cache".into()),
        )
        .expect("create s2");
        s2.append_compaction("Compacted earlier cache audit", None, 42_000)
            .expect("s2 compact");
        s2.append_message(
            Message::user("Verify prompt cache hit ratio."),
            None,
            None,
            None,
        )
        .expect("s2 u1");

        let index1 = SqliteSessionIndex::open(&db_path).expect("open db1");
        let report1 = index1
            .rebuild_from_jsonl_dir(&sessions_dir, &pricing)
            .expect("rebuild 1");
        assert_eq!(report1.sessions_indexed, 2);
        assert_eq!(report1.fts_entries_indexed, 4);
        assert_eq!(report1.usage_rows_indexed, 1);
        assert_eq!(report1.skipped_files, 0);

        let sessions_before = index1.list_sessions(10).expect("list1");
        let summary_before = index1.usage_summary(Some("sess-1")).expect("summary1");
        assert!(summary_before.total_cost_usd > 0.0);
        assert!((summary_before.cache_hit_ratio() - 0.8).abs() < 1e-9);
        drop(index1);

        // Simulate catastrophic deletion of index.sqlite + WAL sidecars
        for suffix in ["", "-wal", "-shm"] {
            let candidate = PathBuf::from(format!("{}{suffix}", db_path.display()));
            if candidate.exists() {
                std::fs::remove_file(candidate).expect("remove sqlite file");
            }
        }

        let index2 = SqliteSessionIndex::open(&db_path).expect("reopen fresh db2");
        let report2 = index2
            .rebuild_from_jsonl_dir(&sessions_dir, &pricing)
            .expect("rebuild 2");
        assert_eq!(report2, report1);
        assert_eq!(index2.list_sessions(10).expect("list2"), sessions_before);
        assert_eq!(
            index2.usage_summary(Some("sess-1")).expect("summary2"),
            summary_before
        );
    }

    #[test]
    fn fts5_bm25_ranks_cross_session_hits_and_handles_special_punctuation() {
        let dir = TempDir::new().expect("tempdir");
        let db_path = dir.path().join("index.sqlite");
        let pricing = PricingCatalog::builtin();
        let index = SqliteSessionIndex::open(&db_path).expect("open");

        let mut s1 = JsonlSessionStore::create(
            dir.path().join("s1.jsonl"),
            "sess-bm25-high",
            "/tmp/repo",
            Some("PGID cleanup".into()),
        )
        .expect("s1");
        s1.append_message(
            Message::assistant_text(
                "Process group PGID killpg cleanup prevents orphan processes after timeout.",
            ),
            None,
            None,
            None,
        )
        .expect("msg1");

        let mut s2 = JsonlSessionStore::create(
            dir.path().join("s2.jsonl"),
            "sess-bm25-other",
            "/tmp/repo",
            Some("Crate setup".into()),
        )
        .expect("s2");
        s2.append_message(
            Message::assistant_text("Updated Cargo.toml and orangeagent-spine dependencies."),
            None,
            None,
            None,
        )
        .expect("msg2");

        index.index_session(&s1, &pricing).expect("index s1");
        index.index_session(&s2, &pricing).expect("index s2");

        let hits = index
            .search_memory("PGID orphan", None, 5)
            .expect("search pgid");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].session_id, "sess-bm25-high");
        assert_eq!(hits[0].session_title.as_deref(), Some("PGID cleanup"));
        assert!(hits[0].bm25_score < 0.0);

        // Hyphenated / dotted queries must be quoted safely so FTS5 never treats '-' as NOT column
        let hyphen_hits = index
            .search_memory("orangeagent-spine Cargo.toml", None, 5)
            .expect("safe punctuation query");
        assert_eq!(hyphen_hits.len(), 1);
        assert_eq!(hyphen_hits[0].session_id, "sess-bm25-other");

        let or_hits = index
            .search_memory("PGID OR Cargo.toml", None, 5)
            .expect("OR query");
        assert_eq!(or_hits.len(), 2);

        assert!(matches!(
            index.search_memory("   ", None, 5),
            Err(SqliteIndexError::InvalidQuery(_))
        ));
    }
}
