use async_trait::async_trait;
use orangeagent_core::{ConcurrencyMode, Tool, ToolContext, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::WorkspaceContext;
use crate::workspace::atomic_write_file;

/// Atomic full-file write tool (`ConcurrencyMode::ExclusiveSequential`).
///
/// Creates missing parent directories, writes via a same-directory temporary file plus
/// `rename(2)`, enforces workspace containment, blocks blind overwrites of existing unread
/// files, and refreshes [`WorkspaceContext::read_tracker`] after persisting.
pub struct WriteTool {
    workspace: WorkspaceContext,
}

impl WriteTool {
    pub fn new(workspace: WorkspaceContext) -> Self {
        Self { workspace }
    }
}

#[derive(Debug, Deserialize)]
struct WriteArgs {
    path: String,
    content: String,
}

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Create a new UTF-8 file or atomically overwrite an existing file inside the workspace."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute path within the workspace."
                },
                "content": {
                    "type": "string",
                    "description": "Full UTF-8 file contents to write."
                }
            },
            "required": ["path", "content"]
        })
    }

    fn concurrency_mode(&self) -> ConcurrencyMode {
        ConcurrencyMode::ExclusiveSequential
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> ToolOutput {
        if ctx.cancel.is_cancelled() {
            return ToolOutput::error("write cancelled");
        }

        let parsed: WriteArgs = match serde_json::from_value(args) {
            Ok(v) => v,
            Err(err) => return ToolOutput::error(format!("invalid arguments for `write`: {err}")),
        };

        let workspace = self.workspace.clone();
        match tokio::task::spawn_blocking(move || write_file_sync(&workspace, &parsed)).await {
            Ok(Ok(summary)) => ToolOutput::ok(summary),
            Ok(Err(err)) => ToolOutput::error(err),
            Err(join_err) => ToolOutput::error(format!("write task failed: {join_err}")),
        }
    }
}

fn write_file_sync(workspace: &WorkspaceContext, args: &WriteArgs) -> Result<String, String> {
    let target = workspace.resolve_target_path(&args.path)?;
    let existed = target.exists();

    if existed {
        if target.is_dir() {
            return Err(format!("`{}` is a directory, not a file", args.path));
        }

        let canonical = workspace.resolve_existing_path(&args.path)?;
        if !workspace.read_tracker().has_read(&canonical) {
            return Err(format!(
                "refusing to overwrite existing file `{}` without reading it first; call `read` before overwriting",
                args.path
            ));
        }

        let current_bytes = std::fs::read(&canonical)
            .map_err(|err| format!("failed to inspect existing file `{}`: {err}", args.path))?;
        if !workspace
            .read_tracker()
            .matches_current(&canonical, &current_bytes)
        {
            return Err(format!(
                "file `{}` was modified externally since it was last read; call `read` again before overwriting",
                args.path
            ));
        }
    }

    let bytes = args.content.as_bytes();
    atomic_write_file(&target, bytes)?;

    let canonical = std::fs::canonicalize(&target)
        .map_err(|err| format!("failed to canonicalize written file `{}`: {err}", args.path))?;
    workspace
        .read_tracker()
        .record_read(canonical.clone(), bytes);

    let rel = workspace.display_relative(&canonical);
    let line_count = args.content.lines().count();
    let action = if existed { "overwrote" } else { "created" };
    Ok(format!(
        "wrote {} bytes ({line_count} lines) to `{rel}` ({action})",
        bytes.len()
    ))
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::ReadTool;

    fn dummy_ctx() -> ToolContext {
        ToolContext {
            turn_id: "t1".into(),
            call_id: "c1".into(),
            cancel: CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn write_creates_nested_file_atomically_and_updates_read_tracker() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ws = WorkspaceContext::new(dir.path()).expect("ws");
        let write_tool = WriteTool::new(ws.clone());

        let out = write_tool
            .execute(
                json!({
                    "path": "src/nested/hello.rs",
                    "content": "fn main() {\n    println!(\"hi\");\n}\n"
                }),
                &dummy_ctx(),
            )
            .await;
        assert!(!out.is_error, "unexpected error: {}", out.content);
        assert!(out.content.contains("(created)"));

        let canonical = ws
            .resolve_existing_path("src/nested/hello.rs")
            .expect("canonical");
        let on_disk = std::fs::read_to_string(&canonical).expect("read back");
        assert!(on_disk.contains("println!(\"hi\")"));
        assert!(
            ws.read_tracker()
                .matches_current(&canonical, on_disk.as_bytes())
        );

        // Subsequent overwrite succeeds because WriteTool updated the tracker.
        let out2 = write_tool
            .execute(
                json!({
                    "path": "src/nested/hello.rs",
                    "content": "fn main() {}\n"
                }),
                &dummy_ctx(),
            )
            .await;
        assert!(!out2.is_error, "unexpected error: {}", out2.content);
        assert!(out2.content.contains("(overwrote)"));
    }

    #[tokio::test]
    async fn write_blocks_blind_overwrite_drift_and_workspace_escape() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("existing.txt");
        std::fs::write(&file, "v1\n").expect("seed");

        let ws = WorkspaceContext::new(dir.path()).expect("ws");
        let read_tool = ReadTool::new(ws.clone());
        let write_tool = WriteTool::new(ws);

        // Blind overwrite before `read` must be rejected.
        let blind = write_tool
            .execute(
                json!({ "path": "existing.txt", "content": "v2\n" }),
                &dummy_ctx(),
            )
            .await;
        assert!(blind.is_error);
        assert!(blind.content.contains("without reading it first"));

        // After `read`, simulate external modification -> must reject drift.
        let read_out = read_tool
            .execute(json!({ "path": "existing.txt" }), &dummy_ctx())
            .await;
        assert!(!read_out.is_error);
        std::fs::write(&file, "external change\n").expect("external write");

        let drift = write_tool
            .execute(
                json!({ "path": "existing.txt", "content": "v2\n" }),
                &dummy_ctx(),
            )
            .await;
        assert!(drift.is_error);
        assert!(drift.content.contains("modified externally"));

        // Escape outside workspace must be rejected.
        let escape = write_tool
            .execute(
                json!({ "path": "../../escaped.txt", "content": "nope" }),
                &dummy_ctx(),
            )
            .await;
        assert!(escape.is_error);
    }
}
