use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::ContextError;

/// Default inline byte cap before spilling full tool output to disk (`30 KB`).
pub const DEFAULT_MAX_INLINE_BYTES: usize = 30 * 1024;
/// Default inline line cap before spilling full tool output to disk (`500` lines).
pub const DEFAULT_MAX_INLINE_LINES: usize = 500;
/// Default head lines preserved in the inline pointer preview.
pub const DEFAULT_HEAD_LINES: usize = 80;
/// Default tail lines preserved in the inline pointer preview.
pub const DEFAULT_TAIL_LINES: usize = 80;

const SINGLE_LINE_HEAD_BYTES: usize = 2_048;
const SINGLE_LINE_TAIL_BYTES: usize = 2_048;

/// Policy controlling entry-point Spill-to-Disk of large tool outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpillConfig {
    pub max_inline_bytes: usize,
    pub max_inline_lines: usize,
    pub head_lines: usize,
    pub tail_lines: usize,
}

impl Default for SpillConfig {
    fn default() -> Self {
        Self {
            max_inline_bytes: DEFAULT_MAX_INLINE_BYTES,
            max_inline_lines: DEFAULT_MAX_INLINE_LINES,
            head_lines: DEFAULT_HEAD_LINES,
            tail_lines: DEFAULT_TAIL_LINES,
        }
    }
}

/// Result of passing a tool output through the Spill-to-Disk gate before enqueueing.
///
/// Once spilled and formatted, `inline_content` is immutable for the rest of the
/// conversation, preserving Prompt Cache prefixes 100%.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpillOutcome {
    pub inline_content: String,
    pub spilled_path: Option<PathBuf>,
    pub original_bytes: usize,
    pub original_lines: usize,
}

/// Spill oversized tool outputs (`> max_inline_bytes` or `> max_inline_lines`) atomically
/// to `<artifacts_root>/<safe_turn_id>/<safe_call_id>.txt` and return a bounded
/// head+tail preview with an explicit inspection pointer.
pub fn spill_tool_output(
    artifacts_root: &Path,
    turn_id: &str,
    call_id: &str,
    raw_output: &str,
    config: &SpillConfig,
) -> Result<SpillOutcome, ContextError> {
    let original_bytes = raw_output.len();
    let lines: Vec<&str> = raw_output.lines().collect();
    let original_lines = lines.len();

    if original_bytes <= config.max_inline_bytes && original_lines <= config.max_inline_lines {
        return Ok(SpillOutcome {
            inline_content: raw_output.to_string(),
            spilled_path: None,
            original_bytes,
            original_lines,
        });
    }

    let safe_turn = sanitize_segment(turn_id);
    let safe_call = sanitize_segment(call_id);
    let target_dir = artifacts_root.join(safe_turn);
    fs::create_dir_all(&target_dir)?;

    let target_path = target_dir.join(format!("{safe_call}.txt"));
    atomic_write(&target_path, raw_output.as_bytes())?;

    let preview = build_preview(raw_output, &lines, &target_path, config);
    Ok(SpillOutcome {
        inline_content: preview,
        spilled_path: Some(target_path),
        original_bytes,
        original_lines,
    })
}

fn sanitize_segment(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches('_');
    if trimmed.is_empty() {
        "unknown".to_string()
    } else {
        trimmed.to_string()
    }
}

fn atomic_write(target: &Path, bytes: &[u8]) -> Result<(), ContextError> {
    let file_name = target
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("artifact.txt");
    let tmp_path = target.with_file_name(format!("{file_name}.{}.tmp", std::process::id()));

    let mut file = File::create(&tmp_path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp_path, target)?;
    Ok(())
}

fn build_preview(
    raw_output: &str,
    lines: &[&str],
    target_path: &Path,
    config: &SpillConfig,
) -> String {
    let total_lines = lines.len();
    let total_bytes = raw_output.len();
    let pointer = format!(
        "[Full output saved to {}, use read/grep to inspect]",
        target_path.display()
    );

    if total_lines > config.head_lines + config.tail_lines {
        let head = lines[..config.head_lines].join("\n");
        let tail_start = total_lines.saturating_sub(config.tail_lines);
        let tail = lines[tail_start..].join("\n");
        let omitted = total_lines.saturating_sub(config.head_lines + config.tail_lines);
        return format!(
            "{head}\n... [omitted {omitted} lines / {total_bytes} total bytes] ...\n{tail}\n\n{pointer}"
        );
    }

    let head = utf8_prefix(raw_output, SINGLE_LINE_HEAD_BYTES);
    let tail = utf8_suffix(raw_output, SINGLE_LINE_TAIL_BYTES);
    let omitted_bytes = total_bytes.saturating_sub(head.len() + tail.len());
    format!(
        "{head}\n... [omitted {omitted_bytes} bytes / {total_bytes} total bytes] ...\n{tail}\n\n{pointer}"
    )
}

fn utf8_prefix(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn utf8_suffix(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut start = s.len().saturating_sub(max_bytes);
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    &s[start..]
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn small_output_passes_through_without_creating_artifact() {
        let dir = TempDir::new().expect("tempdir");
        let raw = "line 1\nline 2\nline 3";
        let outcome = spill_tool_output(dir.path(), "t1", "c1", raw, &SpillConfig::default())
            .expect("small pass-through");

        assert_eq!(outcome.inline_content, raw);
        assert!(outcome.spilled_path.is_none());
        assert_eq!(outcome.original_lines, 3);
    }

    #[test]
    fn large_bytes_or_lines_spills_atomically_with_80_head_and_80_tail_pointer() {
        let dir = TempDir::new().expect("tempdir");
        let cfg = SpillConfig::default();

        // Case 1: 600 lines (> 500 line threshold) + hostile path-traversal segments
        let many_lines: String = (1..=600)
            .map(|i| format!("line-{i:04}: payload"))
            .collect::<Vec<_>>()
            .join("\n");
        let outcome = spill_tool_output(dir.path(), "../../turn/1", "../call:9", &many_lines, &cfg)
            .expect("spill many lines");

        let spilled = outcome.spilled_path.expect("spilled file created");
        assert!(spilled.starts_with(dir.path()));
        assert_eq!(
            std::fs::read_to_string(&spilled).expect("read artifact"),
            many_lines
        );
        assert!(outcome.inline_content.contains("line-0080: payload"));
        assert!(!outcome.inline_content.contains("line-0081: payload"));
        assert!(outcome.inline_content.contains("omitted 440 lines"));
        assert!(outcome.inline_content.contains("line-0521: payload"));
        assert!(!outcome.inline_content.contains("line-0520: payload"));
        assert!(outcome.inline_content.contains("use read/grep to inspect"));

        // Case 2: single minified 40KB line with multi-byte UTF-8 characters (> 30KB byte threshold)
        let huge_line = "中".repeat(14_000);
        assert!(huge_line.len() > DEFAULT_MAX_INLINE_BYTES);
        let outcome2 = spill_tool_output(dir.path(), "turn_2", "call_2", &huge_line, &cfg)
            .expect("spill 1-line");
        let spilled2 = outcome2.spilled_path.expect("spilled2");
        assert_eq!(
            std::fs::read_to_string(&spilled2).expect("read artifact2"),
            huge_line
        );
        assert!(outcome2.inline_content.len() < 5_000);
        assert!(outcome2.inline_content.contains("use read/grep to inspect"));
    }
}
