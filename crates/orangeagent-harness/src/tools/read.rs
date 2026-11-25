use std::fmt::Write as _;

use async_trait::async_trait;
use orangeagent_core::{ConcurrencyMode, Tool, ToolContext, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::WorkspaceContext;

const DEFAULT_LIMIT: usize = 200;
const MAX_LIMIT: usize = 2000;
const BINARY_SNIFF_BYTES: usize = 8192;

/// Read-only file inspection tool (`ConcurrencyMode::ReadOnlyParallel`).
///
/// Supports 1-based line `offset`/`limit`, emits line-numbered output with explicit
/// truncation hints, rejects binary files, and records a content hash in
/// [`WorkspaceContext::read_tracker`] so subsequent edits cannot be performed blindly.
pub struct ReadTool {
    workspace: WorkspaceContext,
}

impl ReadTool {
    pub fn new(workspace: WorkspaceContext) -> Self {
        Self { workspace }
    }
}

#[derive(Debug, Deserialize)]
struct ReadArgs {
    path: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read a UTF-8 text file inside the workspace with 1-based line numbers and pagination."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute path within the workspace."
                },
                "offset": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-based starting line number (default: 1)."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_LIMIT,
                    "description": "Maximum number of lines to return (default: 200)."
                }
            },
            "required": ["path"]
        })
    }

    fn concurrency_mode(&self) -> ConcurrencyMode {
        ConcurrencyMode::ReadOnlyParallel
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> ToolOutput {
        if ctx.cancel.is_cancelled() {
            return ToolOutput::error("read cancelled");
        }

        let parsed: ReadArgs = match serde_json::from_value(args) {
            Ok(v) => v,
            Err(err) => return ToolOutput::error(format!("invalid arguments for `read`: {err}")),
        };

        let offset = parsed.offset.unwrap_or(1).max(1);
        let limit = parsed.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
        let workspace = self.workspace.clone();

        match tokio::task::spawn_blocking(move || {
            read_file_sync(&workspace, &parsed.path, offset, limit)
        })
        .await
        {
            Ok(Ok(text)) => ToolOutput::ok(text),
            Ok(Err(err)) => ToolOutput::error(err),
            Err(join_err) => ToolOutput::error(format!("read task failed: {join_err}")),
        }
    }
}

fn read_file_sync(
    workspace: &WorkspaceContext,
    raw_path: &str,
    offset: usize,
    limit: usize,
) -> Result<String, String> {
    let canonical = workspace.resolve_existing_path(raw_path)?;
    if canonical.is_dir() {
        return Err(format!(
            "`{raw_path}` is a directory; use `glob` to list files"
        ));
    }

    let bytes =
        std::fs::read(&canonical).map_err(|err| format!("failed to read `{raw_path}`: {err}"))?;

    let sniff_len = bytes.len().min(BINARY_SNIFF_BYTES);
    if bytes[..sniff_len].contains(&0) {
        return Err(format!(
            "`{raw_path}` appears to be a binary file ({} bytes)",
            bytes.len()
        ));
    }

    let text = String::from_utf8(bytes.clone())
        .map_err(|_| format!("`{raw_path}` is not valid UTF-8 text"))?;

    // Record full-file digest so Step 7 edit/write can verify the file was read first.
    workspace.read_tracker().record_read(canonical, &bytes);

    let lines: Vec<&str> = text.lines().collect();
    let total_lines = lines.len();
    if total_lines == 0 {
        return Ok(String::from("(empty file)"));
    }

    let start_idx = offset.saturating_sub(1);
    if start_idx >= total_lines {
        return Ok(format!(
            "(offset {offset} is past end of file; total lines: {total_lines})"
        ));
    }

    let end_idx = (start_idx + limit).min(total_lines);
    let mut out = String::new();
    for (idx, line) in lines[start_idx..end_idx].iter().enumerate() {
        let line_no = start_idx + idx + 1;
        let _ = writeln!(out, "{line_no:>4} | {line}");
    }

    if end_idx < total_lines {
        let next_offset = end_idx + 1;
        let _ = write!(
            out,
            "... [truncated: showing lines {offset}..={end_idx} of {total_lines}; pass offset={next_offset} to continue]"
        );
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::*;

    fn dummy_ctx() -> ToolContext {
        ToolContext {
            turn_id: "t1".into(),
            call_id: "c1".into(),
            cancel: CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn read_supports_pagination_truncation_and_records_hash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("sample.rs");
        std::fs::write(&file, "line1\nline2\nline3\nline4\nline5\n").expect("write");

        let ws = WorkspaceContext::new(dir.path()).expect("ws");
        let tool = ReadTool::new(ws.clone());

        let out = tool
            .execute(
                json!({ "path": "sample.rs", "offset": 2, "limit": 2 }),
                &dummy_ctx(),
            )
            .await;
        assert!(!out.is_error, "unexpected error: {}", out.content);
        assert!(out.content.contains("   2 | line2"));
        assert!(out.content.contains("   3 | line3"));
        assert!(out.content.contains("pass offset=4 to continue"));

        let canonical = ws.resolve_existing_path("sample.rs").expect("canonical");
        assert!(ws.read_tracker().has_read(&canonical));
    }

    #[tokio::test]
    async fn read_rejects_binary_files_and_workspace_escapes() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("bin.dat"), b"\x00\x01\x02hello").expect("write");

        let ws = WorkspaceContext::new(dir.path()).expect("ws");
        let tool = ReadTool::new(ws);

        let bin_out = tool
            .execute(json!({ "path": "bin.dat" }), &dummy_ctx())
            .await;
        assert!(bin_out.is_error);
        assert!(bin_out.content.contains("binary"));

        let esc_out = tool
            .execute(json!({ "path": "../../etc/passwd" }), &dummy_ctx())
            .await;
        assert!(esc_out.is_error);
    }
}
