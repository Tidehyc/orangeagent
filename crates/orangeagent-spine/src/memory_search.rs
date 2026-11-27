use std::fmt::Write as _;
use std::sync::Arc;

use async_trait::async_trait;
use orangeagent_core::{ConcurrencyMode, Tool, ToolContext, ToolOutput};
use serde_json::{Value, json};

use crate::session::sqlite::SqliteSessionIndex;

const DEFAULT_SEARCH_LIMIT: usize = 8;

/// On-demand BM25 cross-session memory search tool backed by [`SqliteSessionIndex`].
///
/// Keeps historical conversations out of the static system prompt (preserving Prompt Cache)
/// while letting the model query past decisions, errors, and file paths when needed.
#[derive(Debug, Clone)]
pub struct MemorySearchTool {
    index: Arc<SqliteSessionIndex>,
    default_limit: usize,
}

impl MemorySearchTool {
    pub fn new(index: Arc<SqliteSessionIndex>) -> Self {
        Self {
            index,
            default_limit: DEFAULT_SEARCH_LIMIT,
        }
    }

    #[must_use]
    pub fn with_default_limit(mut self, limit: usize) -> Self {
        self.default_limit = limit.clamp(1, 20);
        self
    }
}

#[async_trait]
impl Tool for MemorySearchTool {
    fn name(&self) -> &str {
        "memory_search"
    }

    fn description(&self) -> &str {
        "Search past conversation sessions by keyword using SQLite FTS5 BM25 ranking."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or phrase to search across historical sessions (supports OR)."
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID filter to restrict search to one conversation."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum number of ranked hits to return (default 8)."
                }
            },
            "required": ["query"],
            "additionalProperties": false
        })
    }

    fn concurrency_mode(&self) -> ConcurrencyMode {
        ConcurrencyMode::ReadOnlyParallel
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> ToolOutput {
        if ctx.cancel.is_cancelled() {
            return ToolOutput::error("memory_search cancelled");
        }

        let Some(query) = args.get("query").and_then(Value::as_str) else {
            return ToolOutput::error("missing required string argument `query`");
        };
        let query = query.trim().to_string();
        if query.is_empty() {
            return ToolOutput::error("argument `query` must not be empty");
        }

        let session_filter = args
            .get("session_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned);

        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(self.default_limit)
            .clamp(1, 20);

        let index = Arc::clone(&self.index);
        let search_res = tokio::task::spawn_blocking(move || {
            index.search_memory(&query, session_filter.as_deref(), limit)
        })
        .await;

        let hits = match search_res {
            Ok(Ok(hits)) => hits,
            Ok(Err(err)) => return ToolOutput::error(format!("memory_search failed: {err}")),
            Err(join_err) => {
                return ToolOutput::error(format!("memory_search task panicked: {join_err}"));
            }
        };

        if hits.is_empty() {
            return ToolOutput::ok("No matching historical session entries found.");
        }

        let mut out = String::new();
        let _ = writeln!(out, "Found {} historical hit(s):", hits.len());
        for (idx, hit) in hits.iter().enumerate() {
            let title = hit.session_title.as_deref().unwrap_or("<untitled>");
            let _ = writeln!(
                out,
                "{}. [session={} title={:?} entry={} role={} bm25={:.4}] {}",
                idx + 1,
                hit.session_id,
                title,
                hit.entry_id,
                hit.role,
                hit.bm25_score,
                hit.snippet.replace('\n', " ")
            );
        }
        ToolOutput::ok(out.trim_end())
    }
}

#[cfg(test)]
mod tests {
    use orangeagent_ai::{Message, PricingCatalog};
    use tempfile::TempDir;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::JsonlSessionStore;

    #[tokio::test]
    async fn memory_search_tool_executes_bm25_search_and_validates_arguments() {
        let dir = TempDir::new().expect("tempdir");
        let index = Arc::new(
            SqliteSessionIndex::open(dir.path().join("index.sqlite")).expect("open sqlite"),
        );

        let mut store = JsonlSessionStore::create(
            dir.path().join("s1.jsonl"),
            "sess-tool-1",
            "/tmp/ws",
            Some("Shadow-Git Drift".into()),
        )
        .expect("create store");
        store
            .append_message(
                Message::assistant_text("Implemented shadow-git recovery_drift detection."),
                None,
                None,
                None,
            )
            .expect("append msg");
        index
            .index_session(&store, &PricingCatalog::builtin())
            .expect("index session");

        let tool = MemorySearchTool::new(Arc::clone(&index)).with_default_limit(5);
        assert_eq!(tool.name(), "memory_search");
        assert_eq!(tool.concurrency_mode(), ConcurrencyMode::ReadOnlyParallel);

        let ctx = ToolContext {
            turn_id: "t1".into(),
            call_id: "c1".into(),
            cancel: CancellationToken::new(),
        };

        let missing = tool.execute(json!({}), &ctx).await;
        assert!(missing.is_error);

        let empty = tool.execute(json!({ "query": "   " }), &ctx).await;
        assert!(empty.is_error);

        let ok = tool
            .execute(json!({ "query": "recovery_drift", "limit": 3 }), &ctx)
            .await;
        assert!(!ok.is_error);
        assert!(ok.content.contains("sess-tool-1"));
        assert!(ok.content.contains("Shadow-Git Drift"));
        assert!(ok.content.contains("[recovery_drift]"));

        let none = tool
            .execute(json!({ "query": "nonexistent_symbol_xyz" }), &ctx)
            .await;
        assert!(!none.is_error);
        assert!(none.content.contains("No matching"));
    }
}
