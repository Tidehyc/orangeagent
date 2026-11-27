//! `orangeagent-ai` — Multi-provider LLM wire models, structured errors,
//! streaming protocol adapters, explicit cache planning, pricing, role routing,
//! and deterministic VCR playback.

pub mod cache;
pub mod error;
pub mod pricing;
pub mod providers;
pub mod router;
pub mod types;
pub mod vcr;

pub use cache::{
    CacheBreakpointSlot, CachePlan, CacheTtl, MAX_ANTHROPIC_CACHE_BREAKPOINTS,
    supports_explicit_cache_control,
};
pub use error::{AiError, ErrorCategory, ErrorVerdict};
pub use pricing::{
    CostBreakdown, EMBEDDED_MODELS_DEV_SNAPSHOT, MODELS_DEV_API_URL, ModelPricing, PricingCatalog,
};
pub use providers::{
    AnthropicConfig, AnthropicProvider, EventStream, LlmProvider, OpenAiChatConfig,
    OpenAiChatProvider, stream_from_events,
};
pub use router::{ModelRole, ModelTarget, RoleModelProfile};
pub use types::{
    ContentBlock, FinishReason, Message, ProviderFamily, ProviderRequest, Role, StreamEvent,
    TokenUsage, ToolDefinition, sanitize_for_provider,
};
pub use vcr::{
    VcrCassette, VcrErrorKind, VcrInteraction, VcrItem, VcrProvider, fingerprint_request,
};
