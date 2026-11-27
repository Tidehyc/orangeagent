use std::path::Path;

use orangeagent_ai::{Message, ToolDefinition};
use tokio_util::sync::CancellationToken;

use crate::ContextError;
use crate::compaction::{
    CompactionConfig, CompactionOutcome, CompactionRecord, CompactionSummarizer, compact_if_needed,
    validate_tool_call_closure,
};
use crate::prefix::{PrefixCheck, PrefixGuard};
use crate::spill::{SpillConfig, SpillOutcome, spill_tool_output};
use crate::tokens::estimate_request_tokens;

/// Result of assembling a prompt request through all 3 cache-safe defense layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssembledContext {
    pub messages: Vec<Message>,
    pub prefix_check: PrefixCheck,
    pub compaction_record: Option<CompactionRecord>,
    pub estimated_tokens: u64,
}

/// Unified facade coordinating Spill-to-Disk, Static Prefix Guard, and
/// High-Watermark Atomic Closure Compaction.
#[derive(Debug, Clone)]
pub struct ContextAssembler {
    prefix_guard: PrefixGuard,
    compaction: CompactionConfig,
    spill: SpillConfig,
}

impl ContextAssembler {
    pub fn new(compaction: CompactionConfig, spill: SpillConfig) -> Self {
        Self {
            prefix_guard: PrefixGuard::new(),
            compaction,
            spill,
        }
    }

    pub fn prefix_guard_mut(&mut self) -> &mut PrefixGuard {
        &mut self.prefix_guard
    }

    /// Defense Layer 1: spill oversized tool outputs to disk before entering history.
    pub fn spill_output(
        &self,
        artifacts_root: &Path,
        turn_id: &str,
        call_id: &str,
        raw_output: &str,
    ) -> Result<SpillOutcome, ContextError> {
        spill_tool_output(artifacts_root, turn_id, call_id, raw_output, &self.spill)
    }

    /// Defense Layers 2 & 3: verify static prefix stability and run atomic compaction
    /// only when history reaches the high watermark (or `force_compact == true`).
    #[allow(clippy::too_many_arguments)]
    pub async fn assemble(
        &mut self,
        system: Option<&str>,
        tools: &[ToolDefinition],
        messages: &[Message],
        last_known_input_tokens: Option<u64>,
        force_compact: bool,
        summarizer: &dyn CompactionSummarizer,
        cancel: CancellationToken,
    ) -> Result<AssembledContext, ContextError> {
        let prefix_check = self.prefix_guard.verify(system, tools);
        let prefix_tokens = estimate_request_tokens(system, tools, &[]);

        match compact_if_needed(
            messages,
            prefix_tokens,
            last_known_input_tokens,
            force_compact,
            &self.compaction,
            summarizer,
            cancel,
        )
        .await?
        {
            CompactionOutcome::PassThrough { estimated_tokens } => Ok(AssembledContext {
                messages: messages.to_vec(),
                prefix_check,
                compaction_record: None,
                estimated_tokens,
            }),
            CompactionOutcome::Compacted {
                messages: compacted,
                record,
            } => {
                if let Err(err) = validate_tool_call_closure(&compacted) {
                    return Err(ContextError::ClosureViolation(format!("{err:?}")));
                }
                let estimated_tokens = record.tokens_after;
                Ok(AssembledContext {
                    messages: compacted,
                    prefix_check,
                    compaction_record: Some(record),
                    estimated_tokens,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use orangeagent_ai::{ContentBlock, Role};
    use serde_json::json;

    use super::*;
    use crate::compaction::DeterministicFallbackSummarizer;

    #[tokio::test]
    async fn assembler_passes_through_below_watermark_and_compacts_above_watermark() {
        let mut assembler = ContextAssembler::new(
            CompactionConfig {
                context_window: 2_000,
                watermark_ratio: 0.75,
                protected_tail_ratio: 0.25,
                min_protected_groups: 1,
            },
            SpillConfig::default(),
        );
        let summarizer = DeterministicFallbackSummarizer;
        let system = Some("You are OrangeAgent.");
        let tools = vec![ToolDefinition {
            name: "read".into(),
            description: "Read file".into(),
            parameters: json!({ "type": "object" }),
        }];

        // Turn 1: small history -> PassThrough (0 mutation) & Pinned
        let small = vec![Message::user("hello"), Message::assistant_text("hi")];
        let res1 = assembler
            .assemble(
                system,
                &tools,
                &small,
                None,
                false,
                &summarizer,
                CancellationToken::new(),
            )
            .await
            .expect("assemble small");
        assert!(matches!(res1.prefix_check, PrefixCheck::Pinned(_)));
        assert!(res1.compaction_record.is_none());
        assert_eq!(res1.messages, small);

        // Turn 2: large history above 75% of 2000 (1500 tokens) -> Compacted + Stable prefix
        let mut big = Vec::new();
        for i in 1..=10 {
            let cid = format!("c_{i}");
            big.push(Message::user(format!("step {i}: check file_{i}.rs")));
            big.push(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall {
                    id: cid.clone(),
                    name: "read".into(),
                    arguments: json!({ "path": format!("file_{i}.rs") }),
                }],
            });
            big.push(Message::tool_result(cid, "a".repeat(600), false));
            big.push(Message::assistant_text(format!("done {i}")));
        }

        let res2 = assembler
            .assemble(
                system,
                &tools,
                &big,
                None,
                false,
                &summarizer,
                CancellationToken::new(),
            )
            .await
            .expect("assemble big");
        assert!(matches!(res2.prefix_check, PrefixCheck::Stable(_)));
        let record = res2.compaction_record.expect("compaction triggered");
        assert!(record.compacted_message_count > 0);
        assert!(record.tokens_after < record.tokens_before);
        assert_eq!(validate_tool_call_closure(&res2.messages), Ok(()));
    }
}
