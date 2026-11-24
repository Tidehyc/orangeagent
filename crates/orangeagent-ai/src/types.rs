use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Wire protocol family supported by `orangeagent-ai`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderFamily {
    /// OpenAI Chat Completions compatible APIs (OpenAI, DeepSeek, OpenRouter, Ollama, vLLM).
    OpenAiChat,
    /// OpenAI Responses API (`encrypted_reasoning`, `prompt_cache_key`).
    OpenAiResponses,
    /// Anthropic Messages API (explicit `cache_control`, signed `thinking` blocks).
    Anthropic,
    /// Google Gemini native `streamGenerateContent` (`thoughtSignature`, `cachedContent`).
    Gemini,
}

/// Speaker role of a conversation message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Typed content block inside a [`Message`].
///
/// Using a Rust tagged enum (`#[serde(tag = "type")]`) makes illegal states
/// unrepresentable at compile time — unlike a flat struct full of `Option<...>` fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text content.
    Text { text: String },

    /// Model reasoning / chain-of-thought block.
    ///
    /// Provider-specific cryptographic signatures (Anthropic `signature`, Gemini
    /// `thoughtSignature`, OpenAI Responses `encrypted_content`) must be preserved
    /// verbatim for the same provider and stripped when falling back across providers.
    Thinking {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider: Option<ProviderFamily>,
    },

    /// A tool invocation emitted by the assistant.
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },

    /// Result returned by executing a [`ContentBlock::ToolCall`].
    ToolResult {
        tool_call_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

/// A single turn message composed of one or more [`ContentBlock`] items.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult {
                tool_call_id: tool_call_id.into(),
                content: content.into(),
                is_error,
            }],
        }
    }

    /// Concatenate all [`ContentBlock::Text`] blocks in this message.
    pub fn text_content(&self) -> String {
        let mut out = String::new();
        for block in &self.content {
            if let ContentBlock::Text { text } = block {
                out.push_str(text);
            }
        }
        out
    }
}

/// JSON-Schema tool declaration sent to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Normalized request passed to any [`ProviderFamily`] adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderRequest {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

/// Normalized token accounting for a single LLM call.
///
/// Convention: `input_tokens` represents the **total** prompt input tokens
/// (including `cache_read_tokens` and `cache_write_tokens`), so
/// `cache_hit_ratio = cache_read_tokens / input_tokens`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_read_tokens: u64,
    #[serde(default)]
    pub cache_write_tokens: u64,
    #[serde(default)]
    pub reasoning_tokens: u64,
}

impl TokenUsage {
    /// Prompt cache hit ratio in `[0.0, 1.0]`. Returns `0.0` when `input_tokens == 0`.
    #[allow(clippy::cast_precision_loss)]
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.input_tokens == 0 {
            return 0.0;
        }
        (self.cache_read_tokens as f64 / self.input_tokens as f64).clamp(0.0, 1.0)
    }

    /// Accumulate another call's usage into `self` using saturating arithmetic.
    pub fn accumulate(&mut self, other: &Self) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cache_read_tokens = self
            .cache_read_tokens
            .saturating_add(other.cache_read_tokens);
        self.cache_write_tokens = self
            .cache_write_tokens
            .saturating_add(other.cache_write_tokens);
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(other.reasoning_tokens);
    }
}

/// Why the model stopped generating tokens for this call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
    Other(String),
}

/// Unified streaming delta event shielding upper layers from wire protocol differences.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    ThinkingDelta {
        text: String,
    },
    ThinkingSignature {
        signature: String,
    },
    TextDelta {
        text: String,
    },
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallArgsDelta {
        index: usize,
        json_fragment: String,
    },
    Usage(TokenUsage),
    Done {
        reason: FinishReason,
    },
}

/// Sanitize conversation history before sending it to a target [`ProviderFamily`].
///
/// - Same provider: keeps `Thinking` blocks and their cryptographic signatures verbatim.
/// - Cross-provider fallback (e.g. Anthropic -> OpenAI): strips foreign signatures to
///   prevent HTTP 400 rejections, and drops signature-only empty thinking blocks.
pub fn sanitize_for_provider(messages: &[Message], target: ProviderFamily) -> Vec<Message> {
    messages
        .iter()
        .map(|msg| {
            let content = msg
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Thinking {
                        text,
                        signature,
                        provider,
                    } => {
                        let same_provider = *provider == Some(target);
                        if same_provider {
                            Some(block.clone())
                        } else if text.is_empty() && signature.is_some() {
                            None
                        } else {
                            Some(ContentBlock::Thinking {
                                text: text.clone(),
                                signature: None,
                                provider: None,
                            })
                        }
                    }
                    other => Some(other.clone()),
                })
                .collect();

            Message {
                role: msg.role,
                content,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn message_and_stream_event_serde_roundtrip() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    text: "plan first".into(),
                    signature: Some("sig_abc".into()),
                    provider: Some(ProviderFamily::Anthropic),
                },
                ContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "read".into(),
                    arguments: json!({ "path": "Cargo.toml" }),
                },
            ],
        };

        let raw = serde_json::to_string(&msg).expect("serialize message");
        assert!(raw.contains("\"type\":\"tool_call\""));
        let decoded: Message = serde_json::from_str(&raw).expect("deserialize message");
        assert_eq!(decoded, msg);

        let event = StreamEvent::ToolCallArgsDelta {
            index: 0,
            json_fragment: "{\"path\":".into(),
        };
        let ev_raw = serde_json::to_string(&event).expect("serialize event");
        let ev_decoded: StreamEvent = serde_json::from_str(&ev_raw).expect("deserialize event");
        assert_eq!(ev_decoded, event);
    }

    #[test]
    fn sanitize_preserves_same_provider_and_strips_foreign_signatures() {
        let history = vec![
            Message::user("fix the build"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking {
                        text: "checking Cargo.toml".into(),
                        signature: Some("anth_sig_1".into()),
                        provider: Some(ProviderFamily::Anthropic),
                    },
                    ContentBlock::Thinking {
                        text: String::new(),
                        signature: Some("encrypted_blob".into()),
                        provider: Some(ProviderFamily::OpenAiResponses),
                    },
                    ContentBlock::Text {
                        text: "Let me inspect Cargo.toml.".into(),
                    },
                ],
            },
        ];

        let for_anthropic = sanitize_for_provider(&history, ProviderFamily::Anthropic);
        assert_eq!(for_anthropic[1].content.len(), 2);
        assert_eq!(
            for_anthropic[1].content[0],
            ContentBlock::Thinking {
                text: "checking Cargo.toml".into(),
                signature: Some("anth_sig_1".into()),
                provider: Some(ProviderFamily::Anthropic),
            }
        );

        let for_openai = sanitize_for_provider(&history, ProviderFamily::OpenAiChat);
        assert_eq!(for_openai[1].content.len(), 2);
        assert_eq!(
            for_openai[1].content[0],
            ContentBlock::Thinking {
                text: "checking Cargo.toml".into(),
                signature: None,
                provider: None,
            }
        );
        assert_eq!(for_openai[1].text_content(), "Let me inspect Cargo.toml.");
    }

    #[test]
    fn token_usage_cache_ratio_and_accumulate() {
        let mut total = TokenUsage::default();
        assert!((total.cache_hit_ratio() - 0.0).abs() < f64::EPSILON);

        let turn1 = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 200,
            cache_read_tokens: 850,
            cache_write_tokens: 150,
            reasoning_tokens: 50,
        };
        total.accumulate(&turn1);
        assert!((total.cache_hit_ratio() - 0.85).abs() < 1e-9);
    }
}
