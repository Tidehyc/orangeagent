use std::fmt::Write as _;
use std::time::SystemTime;

use async_trait::async_trait;
use globset::Glob;
use ignore::WalkBuilder;
use orangeagent_core::{ConcurrencyMode, Tool, ToolContext, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::WorkspaceContext;

const DEFAULT_MAX_RESULTS: usize = 200;
const MAX_RESULTS_CAP: usize = 1000;

/// In-process file discovery tool (`ConcurrencyMode::ReadOnlyParallel`).
///
/// Walks the workspace respecting `.gitignore`, filters relative paths with `globset`,
/// and sorts results by modification time descending (most recently touched files first).
pub struct GlobTool {
    workspace: WorkspaceContext,
}

impl GlobTool {
    pub fn new(workspace: WorkspaceContext) -> Self {
        Self { workspace }
    }
}

#[derive(Debug, Deserialize)]
struct GlobArgs {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    max_results: Option<usize>,
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern inside the workspace (respects .gitignore, sorted by mtime desc)."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern matched against workspace-relative paths (e.g. \"**/*.rs\")."
                },
                "path": {
                    "type": "string",
                    "description": "Optional starting subdirectory (default: workspace root)."
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_RESULTS_CAP,
                    "description": "Maximum number of file paths to return (default: 200)."
                }
            },
            "required": ["pattern"]
        })
    }

    fn concurrency_mode(&self) -> ConcurrencyMode {
        ConcurrencyMode::ReadOnlyParallel
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> ToolOutput {
        if ctx.cancel.is_cancelled() {
            return ToolOutput::error("glob cancelled");
        }

        let parsed: GlobArgs = match serde_json::from_value(args) {
            Ok(v) => v,
            Err(err) => return ToolOutput::error(format!("invalid arguments for `glob`: {err}")),
        };

        let workspace = self.workspace.clone();
        let cancel = ctx.cancel.clone();

        match tokio::task::spawn_blocking(move || glob_sync(&workspace, &parsed, &cancel)).await {
            Ok(Ok(text)) => ToolOutput::ok(text),
            Ok(Err(err)) => ToolOutput::error(err),
            Err(join_err) => ToolOutput::error(format!("glob task failed: {join_err}")),
        }
    }
}

fn glob_sync(
    workspace: &WorkspaceContext,
    args: &GlobArgs,
    cancel: &CancellationToken,
) -> Result<String, String> {
    let target = match &args.path {
        Some(raw) => workspace.resolve_existing_path(raw)?,
        None => workspace.root().to_path_buf(),
    };

    let matcher = Glob::new(&args.pattern)
        .map_err(|err| format!("invalid glob pattern `{}`: {err}", args.pattern))?
        .compile_matcher();

    let max_results = args
        .max_results
        .unwrap_or(DEFAULT_MAX_RESULTS)
        .clamp(1, MAX_RESULTS_CAP);

    let mut walker_builder = WalkBuilder::new(&target);
    walker_builder
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .require_git(false);

    let mut entries: Vec<(SystemTime, String)> = Vec::new();

    for entry in walker_builder.build().flatten() {
        if cancel.is_cancelled() {
            return Err("glob cancelled".into());
        }
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }

        let path = entry.path();
        let rel = workspace.display_relative(path);
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if matcher.is_match(&rel) || matcher.is_match(file_name) {
            let mtime = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            entries.push((mtime, rel));
        }
    }

    if entries.is_empty() {
        return Ok(String::from("(no matching files)"));
    }

    // Sort by modification time descending, breaking ties deterministically by path.
    entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let total = entries.len();
    let truncated = total > max_results;
    entries.truncate(max_results);

    let mut out = entries
        .into_iter()
        .map(|(_, path)| path)
        .collect::<Vec<_>>()
        .join("\n");

    if truncated {
        let _ = write!(
            out,
            "\n... [truncated: showing {max_results} of {total} files]"
        );
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn dummy_ctx() -> ToolContext {
        ToolContext {
            turn_id: "t1".into(),
            call_id: "c1".into(),
            cancel: CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn glob_matches_nested_files_respects_gitignore_and_sorts_by_mtime_desc() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("src")).expect("mkdir");
        std::fs::write(dir.path().join(".gitignore"), "target/\n").expect("gitignore");
        std::fs::create_dir_all(dir.path().join("target")).expect("mkdir target");
        std::fs::write(dir.path().join("target/gen.rs"), "// ignored").expect("write ignored");

        let older = dir.path().join("src/older.rs");
        std::fs::write(&older, "fn old() {}").expect("write older");
        std::thread::sleep(Duration::from_millis(20));
        let newer = dir.path().join("src/newer.rs");
        std::fs::write(&newer, "fn new() {}").expect("write newer");

        let ws = WorkspaceContext::new(dir.path()).expect("ws");
        let tool = GlobTool::new(ws);

        let out = tool
            .execute(json!({ "pattern": "**/*.rs" }), &dummy_ctx())
            .await;
        assert!(!out.is_error, "unexpected error: {}", out.content);
        let lines: Vec<&str> = out.content.lines().collect();
        assert_eq!(lines, vec!["src/newer.rs", "src/older.rs"]);
    }
}
