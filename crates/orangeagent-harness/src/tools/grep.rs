use std::fmt::Write as _;

use async_trait::async_trait;
use globset::{Glob, GlobMatcher};
use grep_regex::RegexMatcherBuilder;
use grep_searcher::Searcher;
use grep_searcher::sinks::UTF8;
use ignore::WalkBuilder;
use orangeagent_core::{ConcurrencyMode, Tool, ToolContext, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::WorkspaceContext;

const DEFAULT_MAX_RESULTS: usize = 100;
const MAX_RESULTS_CAP: usize = 500;
const MAX_LINE_PREVIEW_CHARS: usize = 200;

/// In-process ripgrep-powered regex search tool (`ConcurrencyMode::ReadOnlyParallel`).
///
/// Uses `ignore::WalkBuilder` + `grep-searcher` directly inside the Rust binary (no subprocess
/// fork overhead), automatically respecting `.gitignore` and skipping hidden VCS metadata.
pub struct GrepTool {
    workspace: WorkspaceContext,
}

impl GrepTool {
    pub fn new(workspace: WorkspaceContext) -> Self {
        Self { workspace }
    }
}

#[derive(Debug, Deserialize)]
struct GrepArgs {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    glob: Option<String>,
    #[serde(default)]
    case_insensitive: bool,
    #[serde(default)]
    max_results: Option<usize>,
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search workspace files using a regex pattern (respects .gitignore; in-process ripgrep engine)."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for."
                },
                "path": {
                    "type": "string",
                    "description": "Optional subdirectory or file within the workspace (default: workspace root)."
                },
                "glob": {
                    "type": "string",
                    "description": "Optional file glob filter (e.g. \"*.rs\" or \"src/**/*.ts\")."
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Perform case-insensitive matching (default: false)."
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_RESULTS_CAP,
                    "description": "Maximum number of matching lines to return (default: 100)."
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
            return ToolOutput::error("grep cancelled");
        }

        let parsed: GrepArgs = match serde_json::from_value(args) {
            Ok(v) => v,
            Err(err) => return ToolOutput::error(format!("invalid arguments for `grep`: {err}")),
        };

        let workspace = self.workspace.clone();
        let cancel = ctx.cancel.clone();

        match tokio::task::spawn_blocking(move || grep_sync(&workspace, &parsed, &cancel)).await {
            Ok(Ok(text)) => ToolOutput::ok(text),
            Ok(Err(err)) => ToolOutput::error(err),
            Err(join_err) => ToolOutput::error(format!("grep task failed: {join_err}")),
        }
    }
}

fn grep_sync(
    workspace: &WorkspaceContext,
    args: &GrepArgs,
    cancel: &CancellationToken,
) -> Result<String, String> {
    let target = match &args.path {
        Some(raw) => workspace.resolve_existing_path(raw)?,
        None => workspace.root().to_path_buf(),
    };

    let matcher = RegexMatcherBuilder::new()
        .case_insensitive(args.case_insensitive)
        .build(&args.pattern)
        .map_err(|err| format!("invalid regex `{}`: {err}", args.pattern))?;

    let glob_matcher: Option<GlobMatcher> = match &args.glob {
        Some(pat) => Some(
            Glob::new(pat)
                .map_err(|err| format!("invalid glob `{pat}`: {err}"))?
                .compile_matcher(),
        ),
        None => None,
    };

    let max_results = args
        .max_results
        .unwrap_or(DEFAULT_MAX_RESULTS)
        .clamp(1, MAX_RESULTS_CAP);

    let mut searcher = Searcher::new();
    let mut hits: Vec<String> = Vec::new();
    let mut truncated = false;

    let mut walker_builder = WalkBuilder::new(&target);
    walker_builder
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .require_git(false);

    for entry in walker_builder.build().flatten() {
        if cancel.is_cancelled() {
            return Err("grep cancelled".into());
        }
        if hits.len() >= max_results {
            truncated = true;
            break;
        }
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }

        let path = entry.path();
        let rel = workspace.display_relative(path);
        if let Some(gm) = &glob_matcher {
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !gm.is_match(&rel) && !gm.is_match(file_name) {
                continue;
            }
        }

        let _ = searcher.search_path(
            &matcher,
            path,
            UTF8(|line_num, line| {
                if hits.len() >= max_results {
                    truncated = true;
                    return Ok(false);
                }
                let trimmed = line.trim_end_matches(['\r', '\n']);
                let preview = truncate_line(trimmed, MAX_LINE_PREVIEW_CHARS);
                hits.push(format!("{rel}:{line_num}: {preview}"));
                Ok(true)
            }),
        );
    }

    if hits.is_empty() {
        return Ok(String::from("(no matches)"));
    }

    let mut out = hits.join("\n");
    if truncated {
        let _ = write!(
            out,
            "\n... [truncated at {max_results} matches; narrow `path` or `glob` for more]"
        );
    }
    Ok(out)
}

fn truncate_line(line: &str, max_chars: usize) -> String {
    let mut chars = line.chars();
    let prefix: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{prefix}...")
    } else {
        prefix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_ctx() -> ToolContext {
        ToolContext {
            turn_id: "t1".into(),
            call_id: "c1".into(),
            cancel: CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn grep_respects_gitignore_and_filters_by_glob() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(".gitignore"), "ignored.rs\n").expect("gitignore");
        std::fs::write(dir.path().join("main.rs"), "fn alpha() {}\nfn beta() {}\n").expect("main");
        std::fs::write(dir.path().join("notes.txt"), "alpha in notes\n").expect("notes");
        std::fs::write(dir.path().join("ignored.rs"), "fn alpha() {}\n").expect("ignored");

        let ws = WorkspaceContext::new(dir.path()).expect("ws");
        let tool = GrepTool::new(ws);

        let out = tool
            .execute(json!({ "pattern": "alpha", "glob": "*.rs" }), &dummy_ctx())
            .await;
        assert!(!out.is_error, "unexpected error: {}", out.content);
        assert!(out.content.contains("main.rs:1: fn alpha() {}"));
        assert!(!out.content.contains("notes.txt"));
        assert!(!out.content.contains("ignored.rs"));
    }
}
