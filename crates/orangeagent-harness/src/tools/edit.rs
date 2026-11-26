use std::fmt::Write as _;

use async_trait::async_trait;
use orangeagent_core::{ConcurrencyMode, Tool, ToolContext, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};
use similar::TextDiff;

use crate::WorkspaceContext;
use crate::workspace::atomic_write_file;

const BINARY_SNIFF_BYTES: usize = 8192;
const MAX_DIFF_CHARS: usize = 16_000;

/// Atomic multi-hunk file editing tool (`ConcurrencyMode::ExclusiveSequential`).
///
/// Enforces read-before-edit and external-modification detection via
/// [`WorkspaceContext::read_tracker`], validates and applies all hunks in memory before
/// touching disk (all-or-nothing), falls back to line-level whitespace-normalized fuzzy
/// indentation matching when exact matching fails, and returns a Unified Diff.
pub struct EditTool {
    workspace: WorkspaceContext,
}

impl EditTool {
    pub fn new(workspace: WorkspaceContext) -> Self {
        Self { workspace }
    }
}

#[derive(Debug, Deserialize)]
struct EditHunk {
    old_text: String,
    new_text: String,
    #[serde(default)]
    replace_all: bool,
}

#[derive(Debug, Deserialize)]
struct EditArgs {
    path: String,
    #[serde(default)]
    old_text: Option<String>,
    #[serde(default)]
    new_text: Option<String>,
    #[serde(default)]
    replace_all: bool,
    #[serde(default)]
    edits: Vec<EditHunk>,
}

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Atomically apply one or more search-and-replace hunks to an existing workspace file and return a unified diff."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute path within the workspace."
                },
                "old_text": {
                    "type": "string",
                    "description": "Shorthand single-hunk text to find (mutually exclusive with `edits`)."
                },
                "new_text": {
                    "type": "string",
                    "description": "Shorthand single-hunk replacement text (used with `old_text`)."
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence for the shorthand single hunk (default: false)."
                },
                "edits": {
                    "type": "array",
                    "description": "Ordered list of atomic hunks; all hunks must match or no changes are written.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": { "type": "string" },
                            "new_text": { "type": "string" },
                            "replace_all": { "type": "boolean" }
                        },
                        "required": ["old_text", "new_text"]
                    }
                }
            },
            "required": ["path"]
        })
    }

    fn concurrency_mode(&self) -> ConcurrencyMode {
        ConcurrencyMode::ExclusiveSequential
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> ToolOutput {
        if ctx.cancel.is_cancelled() {
            return ToolOutput::error("edit cancelled");
        }

        let parsed: EditArgs = match serde_json::from_value(args) {
            Ok(v) => v,
            Err(err) => return ToolOutput::error(format!("invalid arguments for `edit`: {err}")),
        };

        let (raw_path, hunks) = match resolve_hunks(parsed) {
            Ok(pair) => pair,
            Err(err) => return ToolOutput::error(err),
        };

        let workspace = self.workspace.clone();
        match tokio::task::spawn_blocking(move || edit_file_sync(&workspace, &raw_path, &hunks))
            .await
        {
            Ok(Ok(diff)) => ToolOutput::ok(diff),
            Ok(Err(err)) => ToolOutput::error(err),
            Err(join_err) => ToolOutput::error(format!("edit task failed: {join_err}")),
        }
    }
}

fn resolve_hunks(args: EditArgs) -> Result<(String, Vec<EditHunk>), String> {
    let has_shorthand = args.old_text.is_some() || args.new_text.is_some();
    if !args.edits.is_empty() && has_shorthand {
        return Err("provide either `edits` or top-level `old_text`/`new_text`, not both".into());
    }

    if !args.edits.is_empty() {
        return Ok((args.path, args.edits));
    }

    match (args.old_text, args.new_text) {
        (Some(old_text), Some(new_text)) => Ok((
            args.path,
            vec![EditHunk {
                old_text,
                new_text,
                replace_all: args.replace_all,
            }],
        )),
        _ => {
            Err("`edit` requires either non-empty `edits` or both `old_text` and `new_text`".into())
        }
    }
}

fn edit_file_sync(
    workspace: &WorkspaceContext,
    raw_path: &str,
    hunks: &[EditHunk],
) -> Result<String, String> {
    let canonical = workspace.resolve_existing_path(raw_path)?;
    if canonical.is_dir() {
        return Err(format!("`{raw_path}` is a directory, not a file"));
    }

    let bytes =
        std::fs::read(&canonical).map_err(|err| format!("failed to read `{raw_path}`: {err}"))?;
    let sniff_len = bytes.len().min(BINARY_SNIFF_BYTES);
    if bytes[..sniff_len].contains(&0) {
        return Err(format!("`{raw_path}` appears to be a binary file"));
    }

    if !workspace.read_tracker().has_read(&canonical) {
        return Err(format!(
            "refusing to edit `{raw_path}` before reading it; call `read` first"
        ));
    }
    if !workspace.read_tracker().matches_current(&canonical, &bytes) {
        return Err(format!(
            "file `{raw_path}` changed on disk since it was last read; call `read` again before editing"
        ));
    }

    let original =
        String::from_utf8(bytes).map_err(|_| format!("`{raw_path}` is not valid UTF-8 text"))?;

    let mut working = original.clone();
    let mut fuzzy_hunks = 0usize;

    for (idx, hunk) in hunks.iter().enumerate() {
        let (next, used_fuzzy) = apply_hunk(&working, hunk, idx + 1, raw_path)?;
        working = next;
        if used_fuzzy {
            fuzzy_hunks += 1;
        }
    }

    if working == original {
        return Err(format!("edit produced no changes to `{raw_path}`"));
    }

    atomic_write_file(&canonical, working.as_bytes())?;
    workspace
        .read_tracker()
        .record_read(canonical.clone(), working.as_bytes());

    let rel = workspace.display_relative(&canonical);
    let diff = TextDiff::from_lines(&original, &working)
        .unified_diff()
        .context_radius(3)
        .header(&format!("a/{rel}"), &format!("b/{rel}"))
        .to_string();

    let mut out = format!("edited `{rel}` ({} hunk(s)", hunks.len());
    if fuzzy_hunks > 0 {
        let _ = write!(out, ", {fuzzy_hunks} fuzzy-indent matched");
    }
    out.push_str(")\n");
    out.push_str(&truncate_chars(&diff, MAX_DIFF_CHARS));
    Ok(out)
}

fn apply_hunk(
    source: &str,
    hunk: &EditHunk,
    hunk_no: usize,
    raw_path: &str,
) -> Result<(String, bool), String> {
    if hunk.old_text.is_empty() {
        return Err(format!("hunk #{hunk_no} has empty `old_text`"));
    }

    let exact_positions: Vec<usize> = source
        .match_indices(&hunk.old_text)
        .map(|(pos, _)| pos)
        .collect();

    if !exact_positions.is_empty() {
        if exact_positions.len() > 1 && !hunk.replace_all {
            return Err(format!(
                "hunk #{hunk_no} matched {} locations in `{raw_path}`; include more context or set replace_all=true",
                exact_positions.len()
            ));
        }

        let updated = if hunk.replace_all {
            source.replace(&hunk.old_text, &hunk.new_text)
        } else {
            source.replacen(&hunk.old_text, &hunk.new_text, 1)
        };
        return Ok((updated, false));
    }

    apply_fuzzy_indent_hunk(source, hunk, hunk_no, raw_path).map(|updated| (updated, true))
}

#[derive(Debug, Clone, Copy)]
struct LineSpan<'a> {
    start: usize,
    end: usize,
    body: &'a str,
    ending: &'a str,
}

fn split_lines_with_endings(text: &str) -> Vec<LineSpan<'_>> {
    let mut spans = Vec::new();
    let mut offset = 0usize;
    for chunk in text.split_inclusive('\n') {
        let start = offset;
        let end = start + chunk.len();
        offset = end;
        let (body, ending) = if let Some(s) = chunk.strip_suffix("\r\n") {
            (s, "\r\n")
        } else if let Some(s) = chunk.strip_suffix('\n') {
            (s, "\n")
        } else {
            (chunk, "")
        };
        spans.push(LineSpan {
            start,
            end,
            body,
            ending,
        });
    }
    spans
}

fn leading_whitespace(line: &str) -> &str {
    let trimmed_len = line.trim_start().len();
    &line[..line.len() - trimmed_len]
}

fn apply_fuzzy_indent_hunk(
    source: &str,
    hunk: &EditHunk,
    hunk_no: usize,
    raw_path: &str,
) -> Result<String, String> {
    let file_lines = split_lines_with_endings(source);
    let old_lines = split_lines_with_endings(&hunk.old_text);
    if old_lines.is_empty() || old_lines.len() > file_lines.len() {
        return Err(format!(
            "hunk #{hunk_no} not found in `{raw_path}` (exact and fuzzy-indent matching both failed)"
        ));
    }

    let window_len = old_lines.len();
    let mut matched_starts = Vec::new();
    let mut idx = 0usize;
    while idx + window_len <= file_lines.len() {
        let window_matches =
            (0..window_len).all(|j| file_lines[idx + j].body.trim() == old_lines[j].body.trim());
        if window_matches {
            matched_starts.push(idx);
            idx += window_len;
        } else {
            idx += 1;
        }
    }

    if matched_starts.is_empty() {
        return Err(format!(
            "hunk #{hunk_no} not found in `{raw_path}` (exact and fuzzy-indent matching both failed)"
        ));
    }
    if matched_starts.len() > 1 && !hunk.replace_all {
        return Err(format!(
            "hunk #{hunk_no} fuzzy-matched {} locations in `{raw_path}`; include more context or set replace_all=true",
            matched_starts.len()
        ));
    }

    let old_base_indent = old_lines
        .iter()
        .find(|l| !l.body.trim().is_empty())
        .map_or("", |l| leading_whitespace(l.body));
    let old_trailing_nl = hunk.old_text.ends_with('\n');
    let new_lines = split_lines_with_endings(&hunk.new_text);

    let mut updated = source.to_owned();
    for &start_idx in matched_starts.iter().rev() {
        let window = &file_lines[start_idx..start_idx + window_len];
        let file_base_indent = window
            .iter()
            .find(|l| !l.body.trim().is_empty())
            .map_or("", |l| leading_whitespace(l.body));
        let eol = window
            .iter()
            .find(|l| !l.ending.is_empty())
            .map_or("\n", |l| l.ending);

        let replacement = reindent_replacement(
            &new_lines,
            old_base_indent,
            file_base_indent,
            eol,
            old_trailing_nl,
        );

        let byte_start = window[0].start;
        let last = window[window_len - 1];
        let byte_end = if old_trailing_nl {
            last.end
        } else {
            last.start + last.body.len()
        };
        updated.replace_range(byte_start..byte_end, &replacement);
    }

    Ok(updated)
}

fn reindent_replacement(
    new_lines: &[LineSpan<'_>],
    old_base_indent: &str,
    file_base_indent: &str,
    eol: &str,
    trailing_eol: bool,
) -> String {
    if new_lines.is_empty() {
        return String::new();
    }

    let mut reindented = Vec::with_capacity(new_lines.len());
    for line in new_lines {
        if line.body.trim().is_empty() {
            reindented.push(String::new());
        } else if let Some(rest) = line.body.strip_prefix(old_base_indent) {
            reindented.push(format!("{file_base_indent}{rest}"));
        } else {
            reindented.push(format!("{file_base_indent}{}", line.body.trim_start()));
        }
    }

    let mut out = reindented.join(eol);
    if trailing_eol {
        out.push_str(eol);
    }
    out
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let prefix: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{prefix}\n... [diff truncated at {max_chars} chars]")
    } else {
        prefix
    }
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
    async fn edit_rejects_unread_and_externally_modified_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("lib.rs");
        std::fs::write(&file, "fn alpha() -> i32 { 1 }\n").expect("seed");

        let ws = WorkspaceContext::new(dir.path()).expect("ws");
        let read_tool = ReadTool::new(ws.clone());
        let edit_tool = EditTool::new(ws);

        // Unread file must be rejected.
        let unread = edit_tool
            .execute(
                json!({
                    "path": "lib.rs",
                    "old_text": "1",
                    "new_text": "2"
                }),
                &dummy_ctx(),
            )
            .await;
        assert!(unread.is_error);
        assert!(unread.content.contains("before reading it"));

        // Read first, then simulate external disk edit -> must reject.
        let _ = read_tool
            .execute(json!({ "path": "lib.rs" }), &dummy_ctx())
            .await;
        std::fs::write(&file, "fn alpha() -> i32 { 99 }\n").expect("external");

        let drifted = edit_tool
            .execute(
                json!({
                    "path": "lib.rs",
                    "old_text": "99",
                    "new_text": "100"
                }),
                &dummy_ctx(),
            )
            .await;
        assert!(drifted.is_error);
        assert!(drifted.content.contains("changed on disk"));
    }

    #[tokio::test]
    async fn edit_applies_multi_hunk_atomically_and_rolls_back_on_partial_failure() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("calc.rs");
        let initial =
            "fn add(a: i32, b: i32) -> i32 { a + b }\nfn sub(a: i32, b: i32) -> i32 { a - b }\n";
        std::fs::write(&file, initial).expect("seed");

        let ws = WorkspaceContext::new(dir.path()).expect("ws");
        let read_tool = ReadTool::new(ws.clone());
        let edit_tool = EditTool::new(ws);

        let _ = read_tool
            .execute(json!({ "path": "calc.rs" }), &dummy_ctx())
            .await;

        // Second hunk fails -> first hunk must NOT be written to disk.
        let failed = edit_tool
            .execute(
                json!({
                    "path": "calc.rs",
                    "edits": [
                        { "old_text": "a + b", "new_text": "a.saturating_add(b)" },
                        { "old_text": "NON_EXISTENT", "new_text": "oops" }
                    ]
                }),
                &dummy_ctx(),
            )
            .await;
        assert!(failed.is_error);
        assert_eq!(std::fs::read_to_string(&file).expect("read"), initial);

        // Both hunks valid -> atomic apply + unified diff output.
        let ok = edit_tool
            .execute(
                json!({
                    "path": "calc.rs",
                    "edits": [
                        { "old_text": "a + b", "new_text": "a.saturating_add(b)" },
                        { "old_text": "a - b", "new_text": "a.saturating_sub(b)" }
                    ]
                }),
                &dummy_ctx(),
            )
            .await;
        assert!(!ok.is_error, "unexpected error: {}", ok.content);
        assert!(ok.content.contains("--- a/calc.rs"));
        assert!(ok.content.contains("+++ b/calc.rs"));
        assert!(
            ok.content
                .contains("+fn add(a: i32, b: i32) -> i32 { a.saturating_add(b) }")
        );
        assert!(
            ok.content
                .contains("+fn sub(a: i32, b: i32) -> i32 { a.saturating_sub(b) }")
        );
    }

    #[tokio::test]
    async fn edit_supports_fuzzy_indent_matching_and_preserves_file_indentation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("indent.rs");
        std::fs::write(
            &file,
            "fn demo() {\n    if true {\n        let x = 1;\n        let y = 2;\n    }\n}\n",
        )
        .expect("seed");

        let ws = WorkspaceContext::new(dir.path()).expect("ws");
        let read_tool = ReadTool::new(ws.clone());
        let edit_tool = EditTool::new(ws);

        let _ = read_tool
            .execute(json!({ "path": "indent.rs" }), &dummy_ctx())
            .await;

        // Model supplies 2-space indent instead of 8-space indent.
        let out = edit_tool
            .execute(
                json!({
                    "path": "indent.rs",
                    "old_text": "  let x = 1;\n  let y = 2;",
                    "new_text": "  let x = 10;\n    let nested = x + 1;\n  let y = 20;"
                }),
                &dummy_ctx(),
            )
            .await;
        assert!(!out.is_error, "unexpected error: {}", out.content);
        assert!(out.content.contains("fuzzy-indent matched"));

        let updated = std::fs::read_to_string(&file).expect("read back");
        assert!(
            updated.contains(
                "        let x = 10;\n          let nested = x + 1;\n        let y = 20;\n"
            )
        );
    }
}
