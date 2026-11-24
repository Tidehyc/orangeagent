//! `orangeagent-ai` — Multi-provider LLM wire models, structured errors,
//! streaming protocol adapters, and deterministic VCR playback.

pub mod error;
pub mod types;

pub use error::{AiError, ErrorCategory, ErrorVerdict};
pub use types::{
    ContentBlock, FinishReason, Message, ProviderFamily, ProviderRequest, Role, StreamEvent,
    TokenUsage, ToolDefinition, sanitize_for_provider,
};
