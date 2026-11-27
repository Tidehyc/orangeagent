//! `orangeagent-context` — Prompt-Cache-first context engine:
//! 1. Entry-point Spill-to-Disk artifact pointer (>30KB / >500 lines)
//! 2. Static Prefix Contract & Drift Guard
//! 3. 75% High-Watermark Atomic Closure Compaction (0 orphaned `tool_call_id`s)

pub mod assembler;
pub mod compaction;
pub mod error;
pub mod prefix;
pub mod spill;
pub mod tokens;

pub use assembler::{AssembledContext, ContextAssembler};
pub use compaction::{
    ClosureViolation, CompactionConfig, CompactionOutcome, CompactionRecord, CompactionSummarizer,
    CompactionSummary, DEFAULT_PROTECTED_TAIL_RATIO, DEFAULT_WATERMARK_RATIO,
    DeterministicFallbackSummarizer, LlmCompactionSummarizer, compact_if_needed,
    find_atomic_split_index, partition_atomic_groups, validate_tool_call_closure,
};
pub use error::ContextError;
pub use prefix::{
    PrefixCheck, PrefixFingerprint, PrefixGuard, SkillSummary, StaticPrefixSpec,
    attach_dynamic_reminder, detect_volatile_system_prompt,
};
pub use spill::{
    DEFAULT_HEAD_LINES, DEFAULT_MAX_INLINE_BYTES, DEFAULT_MAX_INLINE_LINES, DEFAULT_TAIL_LINES,
    SpillConfig, SpillOutcome, spill_tool_output,
};
pub use tokens::{
    estimate_block_tokens, estimate_message_tokens, estimate_messages_tokens,
    estimate_request_tokens, estimate_text_tokens, estimate_tools_tokens,
};
