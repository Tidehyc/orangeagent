use std::collections::VecDeque;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::{AiError, EventStream, LlmProvider, ProviderFamily, ProviderRequest, StreamEvent};

/// Serializable error payload stored in VCR cassettes for deterministic fault injection
/// (e.g., 429 rate limits, context overflow, or mid-stream network drops).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VcrErrorKind {
    RateLimited {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after_ms: Option<u64>,
        message: String,
    },
    ContextOverflow {
        message: String,
    },
    Http {
        status: u16,
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after_ms: Option<u64>,
    },
    Network {
        message: String,
    },
    Protocol {
        message: String,
    },
}

impl From<VcrErrorKind> for AiError {
    fn from(value: VcrErrorKind) -> Self {
        match value {
            VcrErrorKind::RateLimited {
                retry_after_ms,
                message,
            } => Self::RateLimited {
                retry_after_ms,
                message,
            },
            VcrErrorKind::ContextOverflow { message } => Self::ContextOverflow { message },
            VcrErrorKind::Http {
                status,
                message,
                retry_after_ms,
            } => Self::Http {
                status,
                message,
                retry_after_ms,
            },
            VcrErrorKind::Network { message } => Self::Network(message),
            VcrErrorKind::Protocol { message } => Self::Protocol(message),
        }
    }
}

impl VcrErrorKind {
    fn from_ai_error(err: &AiError) -> Option<Self> {
        match err {
            AiError::RateLimited {
                retry_after_ms,
                message,
            } => Some(Self::RateLimited {
                retry_after_ms: *retry_after_ms,
                message: message.clone(),
            }),
            AiError::ContextOverflow { message } => Some(Self::ContextOverflow {
                message: message.clone(),
            }),
            AiError::Http {
                status,
                message,
                retry_after_ms,
            } => Some(Self::Http {
                status: *status,
                message: message.clone(),
                retry_after_ms: *retry_after_ms,
            }),
            AiError::Network(message) => Some(Self::Network {
                message: message.clone(),
            }),
            AiError::Protocol(message) => Some(Self::Protocol {
                message: message.clone(),
            }),
            AiError::Authentication(_)
            | AiError::InvalidRequest(_)
            | AiError::Cancelled
            | AiError::Vcr(_) => None,
        }
    }
}

/// A single recorded item inside a streaming interaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "item_type", rename_all = "snake_case")]
pub enum VcrItem {
    Event { event: StreamEvent },
    Error { error: VcrErrorKind },
}

/// One complete request/response interaction serialized as a single JSONL line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VcrInteraction {
    pub request_hash: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_error: Option<VcrErrorKind>,
    #[serde(default)]
    pub items: Vec<VcrItem>,
}

/// A sequence of recorded [`VcrInteraction`] entries loaded from or saved to a JSONL cassette.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VcrCassette {
    pub interactions: Vec<VcrInteraction>,
}

impl VcrCassette {
    /// Load interactions from a JSONL cassette file (blank lines are ignored).
    pub fn load_jsonl(path: impl AsRef<Path>) -> Result<Self, AiError> {
        let file = fs::File::open(path.as_ref())
            .map_err(|e| AiError::Vcr(format!("failed to open cassette: {e}")))?;
        let reader = BufReader::new(file);
        let mut interactions = Vec::new();
        for (line_no, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| AiError::Vcr(format!("failed to read cassette: {e}")))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let entry: VcrInteraction = serde_json::from_str(trimmed).map_err(|e| {
                AiError::Vcr(format!(
                    "invalid cassette JSON at line {}: {e}",
                    line_no + 1
                ))
            })?;
            interactions.push(entry);
        }
        Ok(Self { interactions })
    }

    /// Overwrite `path` with all interactions in JSONL format.
    pub fn save_jsonl(&self, path: impl AsRef<Path>) -> Result<(), AiError> {
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)
                .map_err(|e| AiError::Vcr(format!("failed to create cassette dir: {e}")))?;
        }
        let mut file = fs::File::create(path.as_ref())
            .map_err(|e| AiError::Vcr(format!("failed to create cassette: {e}")))?;
        for interaction in &self.interactions {
            let line = serde_json::to_string(interaction)
                .map_err(|e| AiError::Vcr(format!("failed to serialize interaction: {e}")))?;
            writeln!(file, "{line}")
                .map_err(|e| AiError::Vcr(format!("failed to write cassette: {e}")))?;
        }
        Ok(())
    }
}

fn append_interaction_jsonl(path: &Path, interaction: &VcrInteraction) -> Result<(), AiError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| AiError::Vcr(format!("failed to create cassette dir: {e}")))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| AiError::Vcr(format!("failed to open cassette for append: {e}")))?;
    let line = serde_json::to_string(interaction)
        .map_err(|e| AiError::Vcr(format!("failed to serialize interaction: {e}")))?;
    writeln!(file, "{line}")
        .map_err(|e| AiError::Vcr(format!("failed to append cassette: {e}")))?;
    Ok(())
}

/// Compute a stable, platform-independent 64-bit FNV-1a hex fingerprint for a [`ProviderRequest`].
///
/// Unlike `std::collections::hash_map::DefaultHasher` (which is randomized per process),
/// FNV-1a produces identical hashes across runs and machines for deterministic VCR matching.
pub fn fingerprint_request(req: &ProviderRequest) -> String {
    let serialized = serde_json::to_vec(req).unwrap_or_default();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in serialized {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{hash:016x}")
}

enum VcrBackend {
    Replay {
        queue: Mutex<VecDeque<VcrInteraction>>,
        strict_hash: bool,
    },
    Record {
        inner: Arc<dyn LlmProvider>,
        path: PathBuf,
        write_lock: Arc<Mutex<()>>,
    },
}

/// Deterministic record/replay [`LlmProvider`] for zero-cost offline testing.
pub struct VcrProvider {
    family: ProviderFamily,
    backend: VcrBackend,
}

impl VcrProvider {
    /// Build an in-memory replay provider from pre-constructed interactions.
    pub fn from_interactions(
        family: ProviderFamily,
        interactions: Vec<VcrInteraction>,
        strict_hash: bool,
    ) -> Self {
        Self {
            family,
            backend: VcrBackend::Replay {
                queue: Mutex::new(interactions.into()),
                strict_hash,
            },
        }
    }

    /// Load a JSONL cassette file for offline deterministic replay.
    pub fn replay_file(
        family: ProviderFamily,
        path: impl AsRef<Path>,
        strict_hash: bool,
    ) -> Result<Self, AiError> {
        let cassette = VcrCassette::load_jsonl(path)?;
        Ok(Self::from_interactions(
            family,
            cassette.interactions,
            strict_hash,
        ))
    }

    /// Wrap a real (or upstream) [`LlmProvider`] and record every streamed interaction to `path`.
    pub fn record(inner: Arc<dyn LlmProvider>, path: impl Into<PathBuf>) -> Self {
        Self {
            family: inner.family(),
            backend: VcrBackend::Record {
                inner,
                path: path.into(),
                write_lock: Arc::new(Mutex::new(())),
            },
        }
    }
}

fn build_record_tee_stream(
    upstream: EventStream,
    path: PathBuf,
    write_lock: Arc<Mutex<()>>,
    req_hash: String,
    model: String,
) -> EventStream {
    let tee_stream = futures::stream::unfold(
        (
            upstream,
            Vec::new(),
            false,
            path,
            write_lock,
            req_hash,
            model,
        ),
        |(mut upstream, mut recorded, flushed, path, write_lock, req_hash, model)| async move {
            if flushed {
                return None;
            }
            match upstream.next().await {
                Some(Ok(event)) => {
                    recorded.push(VcrItem::Event {
                        event: event.clone(),
                    });
                    Some((
                        Ok(event),
                        (upstream, recorded, false, path, write_lock, req_hash, model),
                    ))
                }
                Some(Err(err)) => {
                    if let Some(vcr_err) = VcrErrorKind::from_ai_error(&err) {
                        recorded.push(VcrItem::Error { error: vcr_err });
                    }
                    let interaction = VcrInteraction {
                        request_hash: req_hash.clone(),
                        model: model.clone(),
                        initial_error: None,
                        items: std::mem::take(&mut recorded),
                    };
                    if let Ok(_guard) = write_lock.lock() {
                        let _ = append_interaction_jsonl(&path, &interaction);
                    }
                    Some((
                        Err(err),
                        (upstream, recorded, true, path, write_lock, req_hash, model),
                    ))
                }
                None => {
                    let interaction = VcrInteraction {
                        request_hash: req_hash,
                        model,
                        initial_error: None,
                        items: recorded,
                    };
                    if let Ok(_guard) = write_lock.lock() {
                        let _ = append_interaction_jsonl(&path, &interaction);
                    }
                    None
                }
            }
        },
    );

    Box::pin(tee_stream)
}

#[async_trait]
impl LlmProvider for VcrProvider {
    fn family(&self) -> ProviderFamily {
        self.family
    }

    async fn stream(
        &self,
        req: ProviderRequest,
        cancel: CancellationToken,
    ) -> Result<EventStream, AiError> {
        let req_hash = fingerprint_request(&req);

        match &self.backend {
            VcrBackend::Replay { queue, strict_hash } => {
                let interaction = {
                    let mut guard = queue
                        .lock()
                        .map_err(|_| AiError::Vcr("vcr replay mutex poisoned".into()))?;
                    guard.pop_front().ok_or_else(|| {
                        AiError::Vcr(format!(
                            "cassette exhausted: no interaction left for hash {req_hash}"
                        ))
                    })?
                };

                if *strict_hash && interaction.request_hash != req_hash {
                    return Err(AiError::Vcr(format!(
                        "request hash mismatch: expected {}, got {req_hash}",
                        interaction.request_hash
                    )));
                }

                if let Some(err) = interaction.initial_error {
                    return Err(err.into());
                }

                let events = interaction
                    .items
                    .into_iter()
                    .map(|item| match item {
                        VcrItem::Event { event } => Ok(event),
                        VcrItem::Error { error } => Err(error.into()),
                    })
                    .collect();

                Ok(crate::providers::stream_from_events(events, cancel))
            }
            VcrBackend::Record {
                inner,
                path,
                write_lock,
            } => {
                let model = req.model.clone();
                let upstream = match inner.stream(req, cancel.clone()).await {
                    Ok(stream) => stream,
                    Err(err) => {
                        if let Some(vcr_err) = VcrErrorKind::from_ai_error(&err) {
                            let interaction = VcrInteraction {
                                request_hash: req_hash,
                                model,
                                initial_error: Some(vcr_err),
                                items: Vec::new(),
                            };
                            if let Ok(_guard) = write_lock.lock() {
                                let _ = append_interaction_jsonl(path, &interaction);
                            }
                        }
                        return Err(err);
                    }
                };

                Ok(build_record_tee_stream(
                    upstream,
                    path.clone(),
                    Arc::clone(write_lock),
                    req_hash,
                    model,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use futures::StreamExt;

    use super::*;
    use crate::{ErrorCategory, FinishReason, Message, TokenUsage};

    fn sample_request(prompt: &str) -> ProviderRequest {
        ProviderRequest {
            model: "deepseek-chat".into(),
            system: Some("You are a concise Rust coding agent.".into()),
            messages: vec![Message::user(prompt)],
            tools: Vec::new(),
            max_tokens: Some(1024),
            temperature: Some(0.0),
        }
    }

    fn temp_cassette_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!("orangeagent_vcr_{name}_{nanos}.jsonl"))
    }

    #[tokio::test]
    async fn record_then_replay_roundtrip_with_strict_hash() {
        let req = sample_request("say hi");
        let hash = fingerprint_request(&req);

        let seed = Arc::new(VcrProvider::from_interactions(
            ProviderFamily::OpenAiChat,
            vec![VcrInteraction {
                request_hash: hash.clone(),
                model: req.model.clone(),
                initial_error: None,
                items: vec![
                    VcrItem::Event {
                        event: StreamEvent::TextDelta {
                            text: "Hello ".into(),
                        },
                    },
                    VcrItem::Event {
                        event: StreamEvent::TextDelta {
                            text: "Rust!".into(),
                        },
                    },
                    VcrItem::Event {
                        event: StreamEvent::Usage(TokenUsage {
                            input_tokens: 100,
                            output_tokens: 12,
                            cache_read_tokens: 80,
                            cache_write_tokens: 20,
                            reasoning_tokens: 0,
                        }),
                    },
                    VcrItem::Event {
                        event: StreamEvent::Done {
                            reason: FinishReason::Stop,
                        },
                    },
                ],
            }],
            true,
        ));

        let cassette_path = temp_cassette_path("roundtrip");
        let recorder = VcrProvider::record(seed, &cassette_path);

        let mut stream = recorder
            .stream(req.clone(), CancellationToken::new())
            .await
            .expect("record stream");
        let mut recorded_events = Vec::new();
        while let Some(item) = stream.next().await {
            recorded_events.push(item.expect("ok event"));
        }
        assert_eq!(recorded_events.len(), 4);

        let replayer = VcrProvider::replay_file(ProviderFamily::OpenAiChat, &cassette_path, true)
            .expect("load cassette");
        let mut replay_stream = replayer
            .stream(req, CancellationToken::new())
            .await
            .expect("replay stream");
        let mut replayed_events = Vec::new();
        while let Some(item) = replay_stream.next().await {
            replayed_events.push(item.expect("ok event"));
        }
        assert_eq!(replayed_events, recorded_events);

        let _ = fs::remove_file(cassette_path);
    }

    #[tokio::test]
    async fn replay_supports_initial_and_midstream_fault_injection() {
        let req = sample_request("trigger errors");
        let hash = fingerprint_request(&req);

        let provider = VcrProvider::from_interactions(
            ProviderFamily::Anthropic,
            vec![
                VcrInteraction {
                    request_hash: hash.clone(),
                    model: req.model.clone(),
                    initial_error: Some(VcrErrorKind::RateLimited {
                        retry_after_ms: Some(500),
                        message: "429 burst".into(),
                    }),
                    items: Vec::new(),
                },
                VcrInteraction {
                    request_hash: hash,
                    model: req.model.clone(),
                    initial_error: None,
                    items: vec![
                        VcrItem::Event {
                            event: StreamEvent::TextDelta {
                                text: "partial".into(),
                            },
                        },
                        VcrItem::Error {
                            error: VcrErrorKind::Network {
                                message: "connection reset by peer".into(),
                            },
                        },
                    ],
                },
            ],
            true,
        );

        let first = provider.stream(req.clone(), CancellationToken::new()).await;
        let Err(err) = first else {
            panic!("expected rate limit error");
        };
        assert_eq!(err.verdict().category, ErrorCategory::RateLimit);
        assert_eq!(err.verdict().retry_after_ms, Some(500));

        let mut second = provider
            .stream(req, CancellationToken::new())
            .await
            .expect("second stream starts");
        assert_eq!(
            second.next().await.expect("first chunk").expect("ok"),
            StreamEvent::TextDelta {
                text: "partial".into(),
            }
        );
        let dropped = second.next().await.expect("error chunk").unwrap_err();
        assert_eq!(dropped.verdict().category, ErrorCategory::TransientNetwork);
    }

    #[tokio::test]
    async fn cancellation_token_aborts_stream_immediately() {
        let req = sample_request("cancel me");
        let provider = VcrProvider::from_interactions(
            ProviderFamily::OpenAiChat,
            vec![VcrInteraction {
                request_hash: fingerprint_request(&req),
                model: req.model.clone(),
                initial_error: None,
                items: vec![
                    VcrItem::Event {
                        event: StreamEvent::TextDelta { text: "one".into() },
                    },
                    VcrItem::Event {
                        event: StreamEvent::TextDelta { text: "two".into() },
                    },
                ],
            }],
            false,
        );

        let cancel = CancellationToken::new();
        let mut stream = provider
            .stream(req, cancel.clone())
            .await
            .expect("start stream");

        assert_eq!(
            stream.next().await.expect("chunk 1").expect("ok"),
            StreamEvent::TextDelta { text: "one".into() }
        );
        cancel.cancel();

        let second = stream.next().await.expect("cancelled item").unwrap_err();
        assert_eq!(second.verdict().category, ErrorCategory::Cancelled);
        assert!(stream.next().await.is_none());
    }
}
