//! `orangeagent-ai` — Multi-provider LLM wire models, structured errors,
//! streaming protocol adapters, and deterministic VCR playback.

pub mod error;
pub mod providers;
pub mod types;
pub mod vcr;

pub use error::{AiError, ErrorCategory, ErrorVerdict};
pub use providers::{EventStream, LlmProvider, stream_from_events};
pub use types::{
    ContentBlock, FinishReason, Message, ProviderFamily, ProviderRequest, Role, StreamEvent,
    TokenUsage, ToolDefinition, sanitize_for_provider,
};
pub use vcr::{
    VcrCassette, VcrErrorKind, VcrInteraction, VcrItem, VcrProvider, fingerprint_request,
};
