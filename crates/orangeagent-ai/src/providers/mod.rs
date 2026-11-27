pub mod anthropic;
pub mod openai_chat;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use tokio_util::sync::CancellationToken;

pub use anthropic::{AnthropicConfig, AnthropicProvider};
pub use openai_chat::{OpenAiChatConfig, OpenAiChatProvider};

use crate::{AiError, ProviderFamily, ProviderRequest, StreamEvent};

/// Boxed, pinned, `Send` async stream of [`StreamEvent`] items.
///
/// Using a type alias keeps [`LlmProvider::stream`] readable while still allowing
/// dynamic dispatch (`dyn LlmProvider`) across heterogeneous HTTP/VCR backends.
pub type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, AiError>> + Send>>;

/// Unified asynchronous streaming contract implemented by every LLM wire adapter
/// (OpenAI Chat, OpenAI Responses, Anthropic, Gemini) as well as [`crate::VcrProvider`].
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Identify which wire protocol family this provider speaks.
    fn family(&self) -> ProviderFamily;

    /// Start a streaming completion request, aborting cooperatively when `cancel` fires.
    async fn stream(
        &self,
        req: ProviderRequest,
        cancel: CancellationToken,
    ) -> Result<EventStream, AiError>;
}

/// Convert an in-memory sequence of events into a cancellation-aware [`EventStream`].
///
/// Checks `cancel` before yielding each item; once cancelled it emits a single
/// `Err(AiError::Cancelled)` and terminates the stream cleanly.
pub fn stream_from_events(
    events: Vec<Result<StreamEvent, AiError>>,
    cancel: CancellationToken,
) -> EventStream {
    let iter = events.into_iter();
    Box::pin(futures::stream::unfold(
        (iter, cancel, false),
        |(mut iter, cancel, terminated)| async move {
            if terminated {
                return None;
            }
            if cancel.is_cancelled() {
                return Some((Err(AiError::Cancelled), (iter, cancel, true)));
            }
            let next = iter.next()?;
            Some((next, (iter, cancel, false)))
        },
    ))
}

/// Parse an integer `Retry-After` HTTP header (in seconds) into milliseconds.
pub(crate) fn parse_retry_after_ms(headers: &HeaderMap) -> Option<u64> {
    let raw = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    let secs: u64 = raw.parse().ok()?;
    Some(secs.saturating_mul(1_000))
}
