use std::collections::HashSet;
use std::fmt::Write as _;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use orangeagent_ai::{
    ContentBlock, LlmProvider, Message, ProviderRequest, Role, StreamEvent, sanitize_for_provider,
};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::ContextError;
use crate::tokens::estimate_messages_tokens;

/// Default high-watermark ratio (`75%`) of the context window before triggering compaction.
pub const DEFAULT_WATERMARK_RATIO: f64 = 0.75;
/// Default protected tail budget (`20%`) of the context window preserved verbatim.
pub const DEFAULT_PROTECTED_TAIL_RATIO: f64 = 0.20;

/// Policy governing when and how much history to compact.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompactionConfig {
    pub context_window: u64,
    pub watermark_ratio: f64,
    pub protected_tail_ratio: f64,
    pub min_protected_groups: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            context_window: 200_000,
            watermark_ratio: DEFAULT_WATERMARK_RATIO,
            protected_tail_ratio: DEFAULT_PROTECTED_TAIL_RATIO,
            min_protected_groups: 1,
        }
    }
}

impl CompactionConfig {
    /// Token threshold at or above which compaction triggers.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn high_watermark_tokens(&self) -> u64 {
        let ratio = self.watermark_ratio.clamp(0.1, 0.98);
        ((self.context_window as f64) * ratio).round() as u64
    }

    /// Target token budget for the verbatim protected tail.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn protected_tail_tokens(&self) -> u64 {
        let ratio = self.protected_tail_ratio.clamp(0.05, 0.60);
        ((self.context_window as f64) * ratio).round() as u64
    }
}

/// Structured summary produced by a [`CompactionSummarizer`] replacing compacted history.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionSummary {
    pub goal: String,
    #[serde(default)]
    pub modified_files: Vec<String>,
    #[serde(default)]
    pub key_decisions: Vec<String>,
    #[serde(default)]
    pub pending_and_errors: Vec<String>,
}

impl CompactionSummary {
    /// Render into a deterministic user-message summary block that anchors the
    /// newly compacted prefix.
    pub fn render_markdown(&self) -> String {
        let mut out = String::from("[Compaction Summary]\n");
        let _ = writeln!(out, "## Current Goal\n{}\n", self.goal.trim());
        write_bullet_section(&mut out, "Modified Files", &self.modified_files);
        write_bullet_section(&mut out, "Key Decisions", &self.key_decisions);
        write_bullet_section(
            &mut out,
            "Pending & Unresolved Errors",
            &self.pending_and_errors,
        );
        out.trim_end().to_string()
    }
}

fn write_bullet_section(out: &mut String, heading: &str, items: &[String]) {
    let _ = writeln!(out, "## {heading}");
    if items.is_empty() {
        let _ = writeln!(out, "- (none)\n");
        return;
    }
    for item in items {
        let _ = writeln!(out, "- {}", item.trim());
    }
    out.push('\n');
}

/// Violation detected when checking `tool_call` / `tool_result` pairing closure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClosureViolation {
    LeadingToolMessage { index: usize },
    OrphanedToolResult { index: usize, tool_call_id: String },
    UnansweredToolCall { index: usize, tool_call_id: String },
}

/// Verify that every [`ContentBlock::ToolResult`] matches an earlier [`ContentBlock::ToolCall`]
/// and every `ToolCall` is answered by a subsequent `ToolResult`.
pub fn validate_tool_call_closure(messages: &[Message]) -> Result<(), ClosureViolation> {
    let mut open_calls: HashSet<String> = HashSet::new();

    for (idx, msg) in messages.iter().enumerate() {
        if idx == 0 && msg.role == Role::Tool {
            return Err(ClosureViolation::LeadingToolMessage { index: idx });
        }
        for block in &msg.content {
            match block {
                ContentBlock::ToolCall { id, .. } => {
                    open_calls.insert(id.clone());
                }
                ContentBlock::ToolResult { tool_call_id, .. } => {
                    if !open_calls.remove(tool_call_id) {
                        return Err(ClosureViolation::OrphanedToolResult {
                            index: idx,
                            tool_call_id: tool_call_id.clone(),
                        });
                    }
                }
                ContentBlock::Text { .. } | ContentBlock::Thinking { .. } => {}
            }
        }
    }

    if let Some(dangling) = open_calls.into_iter().min() {
        return Err(ClosureViolation::UnansweredToolCall {
            index: messages.len().saturating_sub(1),
            tool_call_id: dangling,
        });
    }
    Ok(())
}

/// Partition `messages` into indivisible atomic ranges `[start..end)` such that
/// `Assistant(ToolCall...)` and all of its matching `Tool(ToolResult...)` messages
/// always belong to the exact same range.
///
/// Any range boundary `group.start` is mathematically safe as a compaction split point.
pub fn partition_atomic_groups(messages: &[Message]) -> Vec<Range<usize>> {
    let mut groups: Vec<Range<usize>> = Vec::new();
    let mut open_calls: HashSet<String> = HashSet::new();
    let mut group_start = 0usize;

    for (idx, msg) in messages.iter().enumerate() {
        if open_calls.is_empty()
            && msg.role == Role::Tool
            && let Some(prev) = groups.last_mut()
        {
            prev.end = idx + 1;
            group_start = idx + 1;
            continue;
        }

        for block in &msg.content {
            match block {
                ContentBlock::ToolCall { id, .. } => {
                    open_calls.insert(id.clone());
                }
                ContentBlock::ToolResult { tool_call_id, .. } => {
                    open_calls.remove(tool_call_id);
                }
                ContentBlock::Text { .. } | ContentBlock::Thinking { .. } => {}
            }
        }

        if open_calls.is_empty() {
            groups.push(group_start..idx + 1);
            group_start = idx + 1;
        }
    }

    if group_start < messages.len() {
        groups.push(group_start..messages.len());
    }
    groups
}

/// Find the safe message split index `split_idx` (retaining `messages[split_idx..]`
/// as the verbatim protected tail) by scanning atomic groups backward from the end.
///
/// Returns `None` when fewer than 2 atomic groups exist or when the entire history
/// fits inside `protected_tail_budget`.
pub fn find_atomic_split_index(
    messages: &[Message],
    protected_tail_budget: u64,
    min_protected_groups: usize,
) -> Option<usize> {
    let groups = partition_atomic_groups(messages);
    if groups.len() <= 1 {
        return None;
    }

    let min_groups = min_protected_groups.max(1);
    let mut tail_tokens = 0u64;
    let mut split_idx = messages.len();

    for (protected_count, group) in groups.iter().rev().enumerate() {
        let group_tokens = estimate_messages_tokens(&messages[group.clone()]);
        if protected_count >= min_groups
            && tail_tokens.saturating_add(group_tokens) > protected_tail_budget
        {
            break;
        }
        tail_tokens = tail_tokens.saturating_add(group_tokens);
        split_idx = group.start;
    }

    if split_idx == 0 || split_idx >= messages.len() {
        None
    } else {
        Some(split_idx)
    }
}

/// Pluggable summarizer used to condense compacted history prefix into [`CompactionSummary`].
#[async_trait]
pub trait CompactionSummarizer: Send + Sync {
    async fn summarize(
        &self,
        prefix_messages: &[Message],
        cancel: CancellationToken,
    ) -> Result<CompactionSummary, ContextError>;
}

/// Zero-cost deterministic offline summarizer (useful in tests and fallback paths).
#[derive(Debug, Default, Clone, Copy)]
pub struct DeterministicFallbackSummarizer;

#[async_trait]
impl CompactionSummarizer for DeterministicFallbackSummarizer {
    async fn summarize(
        &self,
        prefix_messages: &[Message],
        cancel: CancellationToken,
    ) -> Result<CompactionSummary, ContextError> {
        if cancel.is_cancelled() {
            return Err(ContextError::Cancelled);
        }

        let goal = prefix_messages
            .iter()
            .find(|m| m.role == Role::User && !m.text_content().trim().is_empty())
            .map_or_else(
                || "Continue prior coding session".to_string(),
                |m| truncate_inline(&m.text_content(), 200),
            );

        let mut files = Vec::new();
        let mut seen_files = HashSet::new();
        let mut decisions = Vec::new();
        let mut errors = Vec::new();

        for msg in prefix_messages {
            for block in &msg.content {
                match block {
                    ContentBlock::ToolCall {
                        name, arguments, ..
                    } if matches!(name.as_str(), "write" | "edit") => {
                        if let Some(path) = arguments.get("path").and_then(|v| v.as_str())
                            && seen_files.insert(path.to_string())
                        {
                            files.push(path.to_string());
                        }
                    }
                    ContentBlock::ToolResult {
                        content,
                        is_error: true,
                        ..
                    } => {
                        errors.push(truncate_inline(content, 160));
                    }
                    ContentBlock::Text { text }
                        if msg.role == Role::Assistant && !text.trim().is_empty() =>
                    {
                        decisions.push(truncate_inline(text, 180));
                    }
                    _ => {}
                }
            }
        }

        if decisions.len() > 5 {
            decisions = decisions.split_off(decisions.len() - 5);
        }
        if errors.len() > 5 {
            errors = errors.split_off(errors.len() - 5);
        }

        Ok(CompactionSummary {
            goal,
            modified_files: files,
            key_decisions: decisions,
            pending_and_errors: errors,
        })
    }
}

/// LLM-backed compaction summarizer routed via [`LlmProvider`] (typically `RoleModelProfile.compactor`).
#[derive(Clone)]
pub struct LlmCompactionSummarizer {
    provider: Arc<dyn LlmProvider>,
    model: String,
}

impl LlmCompactionSummarizer {
    pub fn new(provider: Arc<dyn LlmProvider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

#[async_trait]
impl CompactionSummarizer for LlmCompactionSummarizer {
    async fn summarize(
        &self,
        prefix_messages: &[Message],
        cancel: CancellationToken,
    ) -> Result<CompactionSummary, ContextError> {
        let transcript = render_prefix_for_summary(prefix_messages);
        let prompt = format!(
            "Summarize the following completed conversation prefix into strict JSON with keys \
             `goal` (string), `modified_files` (string[]), `key_decisions` (string[]), \
             and `pending_and_errors` (string[]).\n\n{transcript}"
        );
        let sanitized = sanitize_for_provider(&[Message::user(prompt)], self.provider.family());
        let req = ProviderRequest {
            model: self.model.clone(),
            system: Some(
                "You compress coding agent transcripts accurately without inventing file paths."
                    .to_string(),
            ),
            messages: sanitized,
            tools: Vec::new(),
            max_tokens: Some(1_024),
            temperature: Some(0.0),
        };

        let mut stream = self.provider.stream(req, cancel).await?;
        let mut raw_text = String::new();
        while let Some(item) = stream.next().await {
            if let StreamEvent::TextDelta { text } = item? {
                raw_text.push_str(&text);
            }
        }
        Ok(parse_summary_response(&raw_text))
    }
}

/// Telemetry emitted when compaction rewrites the prefix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionRecord {
    pub summary: CompactionSummary,
    pub compacted_message_count: usize,
    pub kept_message_count: usize,
    pub tokens_before: u64,
    pub tokens_after: u64,
}

/// Result of evaluating [`compact_if_needed`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionOutcome {
    /// Below high watermark — zero mutation pass-through preserving 100% of Prompt Cache.
    PassThrough { estimated_tokens: u64 },
    /// Above watermark (or forced) — prefix replaced with summary + atomic protected tail.
    Compacted {
        messages: Vec<Message>,
        record: CompactionRecord,
    },
}

/// Compact `messages` only when token usage hits the `75%` high watermark (or `force == true`).
pub async fn compact_if_needed(
    messages: &[Message],
    system_and_tools_tokens: u64,
    last_known_input_tokens: Option<u64>,
    force: bool,
    config: &CompactionConfig,
    summarizer: &dyn CompactionSummarizer,
    cancel: CancellationToken,
) -> Result<CompactionOutcome, ContextError> {
    let heuristic_total =
        system_and_tools_tokens.saturating_add(estimate_messages_tokens(messages));
    let effective_tokens =
        last_known_input_tokens.map_or(heuristic_total, |known| known.max(heuristic_total));

    if !force && effective_tokens < config.high_watermark_tokens() {
        return Ok(CompactionOutcome::PassThrough {
            estimated_tokens: effective_tokens,
        });
    }

    let Some(split_idx) = find_atomic_split_index(
        messages,
        config.protected_tail_tokens(),
        config.min_protected_groups,
    ) else {
        return Ok(CompactionOutcome::PassThrough {
            estimated_tokens: effective_tokens,
        });
    };

    let prefix = &messages[..split_idx];
    let kept_tail = &messages[split_idx..];

    if let Err(err) = validate_tool_call_closure(kept_tail) {
        return Err(ContextError::ClosureViolation(format!("{err:?}")));
    }

    let summary = summarizer.summarize(prefix, cancel).await?;
    let mut compacted = Vec::with_capacity(1 + kept_tail.len());
    compacted.push(Message::user(summary.render_markdown()));
    compacted.extend(kept_tail.iter().cloned());

    if let Err(err) = validate_tool_call_closure(&compacted) {
        return Err(ContextError::ClosureViolation(format!("{err:?}")));
    }

    let tokens_after = system_and_tools_tokens.saturating_add(estimate_messages_tokens(&compacted));
    Ok(CompactionOutcome::Compacted {
        messages: compacted,
        record: CompactionRecord {
            summary,
            compacted_message_count: prefix.len(),
            kept_message_count: kept_tail.len(),
            tokens_before: effective_tokens,
            tokens_after,
        },
    })
}

fn render_prefix_for_summary(prefix: &[Message]) -> String {
    let mut out = String::new();
    for msg in prefix {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        let _ = writeln!(out, "[{role}]");
        for block in &msg.content {
            match block {
                ContentBlock::Text { text } => {
                    let _ = writeln!(out, "{}", truncate_inline(text, 600));
                }
                ContentBlock::ToolCall {
                    name, arguments, ..
                } => {
                    let _ = writeln!(out, "tool_call {name}({arguments})");
                }
                ContentBlock::ToolResult {
                    content, is_error, ..
                } => {
                    let _ = writeln!(
                        out,
                        "tool_result(err={is_error}): {}",
                        truncate_inline(content, 400)
                    );
                }
                ContentBlock::Thinking { .. } => {}
            }
        }
    }
    out
}

fn parse_summary_response(raw: &str) -> CompactionSummary {
    let trimmed = raw.trim();
    let stripped = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|s| s.strip_suffix("```"))
        .map_or(trimmed, str::trim);

    if let Ok(parsed) = serde_json::from_str::<CompactionSummary>(stripped)
        && !parsed.goal.trim().is_empty()
    {
        return parsed;
    }

    CompactionSummary {
        goal: if trimmed.is_empty() {
            "Compacted prior conversation".to_string()
        } else {
            trimmed.to_string()
        },
        modified_files: Vec::new(),
        key_decisions: Vec::new(),
        pending_and_errors: Vec::new(),
    }
}

fn truncate_inline(text: &str, max_chars: usize) -> String {
    let flat = text.replace('\n', " ");
    let mut chars = flat.chars();
    let head: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{head}...")
    } else {
        head
    }
}

#[cfg(test)]
mod tests {
    use orangeagent_ai::{
        FinishReason, ProviderFamily, TokenUsage, VcrInteraction, VcrItem, VcrProvider,
        fingerprint_request,
    };
    use serde_json::json;

    use super::*;

    fn build_tool_round(round: usize, payload_repeat: usize) -> Vec<Message> {
        let call_a = format!("call_{round}_a");
        let call_b = format!("call_{round}_b");
        let filler = "x".repeat(payload_repeat);
        vec![
            Message::user(format!("round {round}: inspect file_{round}.rs")),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::ToolCall {
                        id: call_a.clone(),
                        name: "read".into(),
                        arguments: json!({ "path": format!("src/mod_{round}.rs") }),
                    },
                    ContentBlock::ToolCall {
                        id: call_b.clone(),
                        name: "edit".into(),
                        arguments: json!({ "path": format!("src/mod_{round}.rs") }),
                    },
                ],
            },
            Message::tool_result(call_a, format!("contents-{round}: {filler}"), false),
            Message::tool_result(call_b, format!("edited src/mod_{round}.rs"), false),
            Message::assistant_text(format!("Completed round {round} edits.")),
        ]
    }

    #[test]
    fn atomic_closure_never_splits_parallel_tool_calls_or_orphans_tool_results() {
        let mut history = Vec::new();
        history.extend(build_tool_round(1, 200));
        history.extend(build_tool_round(2, 200));
        history.extend(build_tool_round(3, 200));

        assert_eq!(validate_tool_call_closure(&history), Ok(()));
        let groups = partition_atomic_groups(&history);
        // Each round produces 3 atomic groups: User, Assistant(2x ToolCall)+2x ToolResult, Assistant(final)
        assert_eq!(groups.len(), 9);
        assert_eq!(groups[1], 1..4);

        // Scan across a wide range of protected-tail budgets: EVERY split index keeps closure valid
        for budget in [1_u64, 50, 120, 250, 400, 800, 5_000] {
            if let Some(split_idx) = find_atomic_split_index(&history, budget, 1) {
                assert_ne!(
                    history[split_idx].role,
                    Role::Tool,
                    "split_idx={split_idx} must never start on Role::Tool"
                );
                assert_eq!(
                    validate_tool_call_closure(&history[split_idx..]),
                    Ok(()),
                    "split_idx={split_idx} at budget={budget} broke closure"
                );
            }
        }
    }

    #[tokio::test]
    async fn fifty_round_stress_compaction_produces_zero_orphaned_tool_call_ids() {
        let cfg = CompactionConfig {
            context_window: 8_000,
            watermark_ratio: 0.75,
            protected_tail_ratio: 0.25,
            min_protected_groups: 2,
        };
        let summarizer = DeterministicFallbackSummarizer;
        let mut messages: Vec<Message> = Vec::new();
        let mut compaction_count = 0usize;

        for round in 1..=50 {
            messages.extend(build_tool_round(round, 600));
            let outcome = compact_if_needed(
                &messages,
                200,
                None,
                false,
                &cfg,
                &summarizer,
                CancellationToken::new(),
            )
            .await
            .expect("compaction succeeds");

            if let CompactionOutcome::Compacted {
                messages: next_msgs,
                record,
            } = outcome
            {
                compaction_count += 1;
                assert!(record.tokens_after < record.tokens_before);
                assert_eq!(
                    validate_tool_call_closure(&next_msgs),
                    Ok(()),
                    "compaction #{compaction_count} at round {round} produced closure violation"
                );
                messages = next_msgs;
            } else {
                assert_eq!(validate_tool_call_closure(&messages), Ok(()));
            }
        }

        assert!(
            compaction_count >= 3,
            "50-round stress test should exercise compaction multiple times (got {compaction_count})"
        );
        assert_eq!(validate_tool_call_closure(&messages), Ok(()));
    }

    #[tokio::test]
    async fn llm_compaction_summarizer_works_with_vcr_provider() {
        let summary_json = r#"```json
{
  "goal": "Implement orangeagent-context compaction",
  "modified_files": ["crates/orangeagent-context/src/compaction.rs"],
  "key_decisions": ["Use atomic tool_call closure alignment"],
  "pending_and_errors": []
}
```"#;
        let interaction = VcrInteraction {
            request_hash: fingerprint_request(&ProviderRequest {
                model: "claude-haiku-4-5".into(),
                system: None,
                messages: vec![Message::user("ignored when strict_hash=false")],
                tools: Vec::new(),
                max_tokens: None,
                temperature: None,
            }),
            model: "claude-haiku-4-5".into(),
            initial_error: None,
            items: vec![
                VcrItem::Event {
                    event: StreamEvent::TextDelta {
                        text: summary_json.into(),
                    },
                },
                VcrItem::Event {
                    event: StreamEvent::Usage(TokenUsage {
                        input_tokens: 500,
                        output_tokens: 60,
                        ..TokenUsage::default()
                    }),
                },
                VcrItem::Event {
                    event: StreamEvent::Done {
                        reason: FinishReason::Stop,
                    },
                },
            ],
        };
        let vcr = Arc::new(VcrProvider::from_interactions(
            ProviderFamily::Anthropic,
            vec![interaction],
            false,
        ));

        let summarizer = LlmCompactionSummarizer::new(vcr, "claude-haiku-4-5");
        let prefix = build_tool_round(1, 20);
        let summary = summarizer
            .summarize(&prefix, CancellationToken::new())
            .await
            .expect("vcr summarize");

        assert_eq!(summary.goal, "Implement orangeagent-context compaction");
        assert_eq!(
            summary.modified_files,
            vec!["crates/orangeagent-context/src/compaction.rs"]
        );
        assert!(summary.render_markdown().contains("[Compaction Summary]"));
    }
}
