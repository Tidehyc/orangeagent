use std::collections::{BTreeMap, VecDeque};

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::{
    AiError, CachePlan, CacheTtl, ContentBlock, EventStream, FinishReason, LlmProvider, Message,
    ProviderFamily, ProviderRequest, Role, StreamEvent, TokenUsage, sanitize_for_provider,
};

use super::openai_chat::SseLineDecoder;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_API_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 8_192;
const EXTENDED_CACHE_BETA: &str = "extended-cache-ttl-2025-04-11";

/// Configuration for the native Anthropic Messages streaming adapter.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// Base URL without `/v1/messages` (defaults to `https://api.anthropic.com`).
    pub base_url: String,
    /// `x-api-key` credential.
    pub api_key: String,
    /// `anthropic-version` header (defaults to `"2023-06-01"`).
    pub api_version: String,
    /// Fallback `max_tokens` when `ProviderRequest::max_tokens` is `None`.
    pub default_max_tokens: u32,
    /// Whether to emit up to 4 explicit `cache_control` breakpoints.
    pub cache_enabled: bool,
    /// Ephemeral cache TTL (`5m` default or `1h` with beta header).
    pub cache_ttl: CacheTtl,
    /// Optional extended thinking token budget (`thinking: { type: "enabled", budget_tokens }`).
    pub thinking_budget_tokens: Option<u32>,
}

impl AnthropicConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            api_version: DEFAULT_API_VERSION.to_string(),
            default_max_tokens: DEFAULT_MAX_TOKENS,
            cache_enabled: true,
            cache_ttl: CacheTtl::default(),
            thinking_budget_tokens: None,
        }
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    #[must_use]
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = api_version.into();
        self
    }

    #[must_use]
    pub fn with_default_max_tokens(mut self, max_tokens: u32) -> Self {
        self.default_max_tokens = max_tokens;
        self
    }

    #[must_use]
    pub fn with_cache_enabled(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    #[must_use]
    pub fn with_cache_ttl(mut self, ttl: CacheTtl) -> Self {
        self.cache_ttl = ttl;
        self
    }

    #[must_use]
    pub fn with_thinking_budget(mut self, budget_tokens: Option<u32>) -> Self {
        self.thinking_budget_tokens = budget_tokens;
        self
    }

    /// Resolve the full `/v1/messages` endpoint URL without duplicating `/v1`.
    pub fn endpoint_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        if base.ends_with("/v1") {
            format!("{base}/messages")
        } else {
            format!("{base}/v1/messages")
        }
    }
}

/// Native streaming client for Anthropic's `/v1/messages` API.
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    config: AnthropicConfig,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Result<Self, AiError> {
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| AiError::Network(format!("failed to build HTTP client: {e}")))?;
        Ok(Self { config, client })
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn family(&self) -> ProviderFamily {
        ProviderFamily::Anthropic
    }

    async fn stream(
        &self,
        req: ProviderRequest,
        cancel: CancellationToken,
    ) -> Result<EventStream, AiError> {
        let body = build_request_body(&req, &self.config);
        let headers = build_headers(&self.config)?;

        let send_fut = self
            .client
            .post(self.config.endpoint_url())
            .headers(headers)
            .json(&body)
            .send();

        let response = tokio::select! {
            () = cancel.cancelled() => return Err(AiError::Cancelled),
            res = send_fut => res.map_err(|e| AiError::Network(e.to_string()))?,
        };

        let status = response.status();
        if !status.is_success() {
            let retry_after_ms = super::parse_retry_after_ms(response.headers());
            let err_body = response.text().await.unwrap_or_default();
            return Err(classify_http_error(
                status.as_u16(),
                &err_body,
                retry_after_ms,
            ));
        }

        let byte_stream = Box::pin(response.bytes_stream());
        Ok(build_anthropic_event_stream(byte_stream, cancel))
    }
}

fn build_headers(config: &AnthropicConfig) -> Result<HeaderMap, AiError> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let key = HeaderValue::from_str(&config.api_key)
        .map_err(|e| AiError::Authentication(format!("invalid API key header: {e}")))?;
    headers.insert(HeaderName::from_static("x-api-key"), key);

    let version = HeaderValue::from_str(&config.api_version)
        .map_err(|e| AiError::InvalidRequest(format!("invalid anthropic-version header: {e}")))?;
    headers.insert(HeaderName::from_static("anthropic-version"), version);

    if config.cache_enabled && config.cache_ttl.requires_extended_ttl_beta() {
        headers.insert(
            HeaderName::from_static("anthropic-beta"),
            HeaderValue::from_static(EXTENDED_CACHE_BETA),
        );
    }
    Ok(headers)
}

#[derive(Debug)]
struct WireMessage {
    role: &'static str,
    blocks: Vec<Value>,
}

/// Convert a normalized [`ProviderRequest`] into an Anthropic `/v1/messages` JSON payload
/// with explicit 4-breakpoint `cache_control` placement and signed thinking preservation.
pub fn build_request_body(req: &ProviderRequest, config: &AnthropicConfig) -> Value {
    let sanitized = sanitize_for_provider(&req.messages, ProviderFamily::Anthropic);
    let mut system_blocks = Vec::new();
    if let Some(system) = &req.system
        && !system.is_empty()
    {
        system_blocks.push(json!({
            "type": "text",
            "text": system,
        }));
    }

    let mut wire_messages: Vec<WireMessage> = Vec::with_capacity(sanitized.len());
    for msg in &sanitized {
        if msg.role == Role::System {
            let text = msg.text_content();
            if !text.is_empty() {
                system_blocks.push(json!({
                    "type": "text",
                    "text": text,
                }));
            }
            continue;
        }
        let (role, blocks) = convert_message_blocks(msg);
        if blocks.is_empty() {
            continue;
        }
        if let Some(prev) = wire_messages.last_mut()
            && prev.role == role
        {
            prev.blocks.extend(blocks);
        } else {
            wire_messages.push(WireMessage { role, blocks });
        }
    }

    let mut wire_tools: Vec<Value> = req
        .tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        })
        .collect();

    let cacheable_indices: Vec<usize> = wire_messages
        .iter()
        .enumerate()
        .filter_map(|(idx, wm)| wm.blocks.iter().any(is_cacheable_block).then_some(idx))
        .collect();

    let plan = if config.cache_enabled {
        CachePlan::for_anthropic(
            !wire_tools.is_empty(),
            !system_blocks.is_empty(),
            &cacheable_indices,
            config.cache_ttl,
        )
    } else {
        CachePlan::disabled()
    };

    apply_cache_plan(
        &plan,
        &mut wire_tools,
        &mut system_blocks,
        &mut wire_messages,
    );

    assemble_payload(req, config, wire_tools, system_blocks, wire_messages)
}

fn convert_message_blocks(msg: &Message) -> (&'static str, Vec<Value>) {
    match msg.role {
        Role::User => {
            let blocks = msg
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } if !text.is_empty() => Some(json!({
                        "type": "text",
                        "text": text,
                    })),
                    ContentBlock::ToolResult {
                        tool_call_id,
                        content,
                        is_error,
                    } => Some(tool_result_json(tool_call_id, content, *is_error)),
                    _ => None,
                })
                .collect();
            ("user", blocks)
        }
        Role::Tool => {
            let blocks = msg
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolResult {
                        tool_call_id,
                        content,
                        is_error,
                    } => Some(tool_result_json(tool_call_id, content, *is_error)),
                    _ => None,
                })
                .collect();
            ("user", blocks)
        }
        Role::Assistant => {
            let blocks = msg
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Thinking {
                        text,
                        signature: Some(sig),
                        provider: Some(ProviderFamily::Anthropic),
                    } => {
                        if text.is_empty() {
                            Some(json!({
                                "type": "redacted_thinking",
                                "data": sig,
                            }))
                        } else {
                            Some(json!({
                                "type": "thinking",
                                "thinking": text,
                                "signature": sig,
                            }))
                        }
                    }
                    ContentBlock::Text { text } if !text.is_empty() => Some(json!({
                        "type": "text",
                        "text": text,
                    })),
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } => Some(json!({
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": arguments,
                    })),
                    _ => None,
                })
                .collect();
            ("assistant", blocks)
        }
        Role::System => ("user", Vec::new()),
    }
}

fn tool_result_json(tool_call_id: &str, content: &str, is_error: bool) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("tool_result"));
    obj.insert("tool_use_id".into(), json!(tool_call_id));
    obj.insert("content".into(), json!(content));
    if is_error {
        obj.insert("is_error".into(), json!(true));
    }
    Value::Object(obj)
}

fn is_cacheable_block(block: &Value) -> bool {
    let kind = block.get("type").and_then(Value::as_str);
    !matches!(kind, Some("thinking" | "redacted_thinking"))
}

fn apply_cache_plan(
    plan: &CachePlan,
    wire_tools: &mut [Value],
    system_blocks: &mut [Value],
    wire_messages: &mut [WireMessage],
) {
    if plan.total_breakpoints() == 0 {
        return;
    }
    let marker = plan.ttl.to_anthropic_json();

    if plan.mark_last_tool
        && let Some(last_tool) = wire_tools.last_mut()
    {
        attach_cache_control(last_tool, &marker);
    }

    if plan.mark_system
        && let Some(last_sys) = system_blocks.last_mut()
    {
        attach_cache_control(last_sys, &marker);
    }

    for &msg_idx in &plan.message_indices {
        if let Some(msg) = wire_messages.get_mut(msg_idx)
            && let Some(block) = msg.blocks.iter_mut().rev().find(|b| is_cacheable_block(b))
        {
            attach_cache_control(block, &marker);
        }
    }
}

fn attach_cache_control(target: &mut Value, marker: &Value) {
    if let Value::Object(map) = target {
        map.insert("cache_control".into(), marker.clone());
    }
}

fn assemble_payload(
    req: &ProviderRequest,
    config: &AnthropicConfig,
    wire_tools: Vec<Value>,
    system_blocks: Vec<Value>,
    wire_messages: Vec<WireMessage>,
) -> Value {
    let mut max_tokens = req.max_tokens.unwrap_or(config.default_max_tokens);
    if let Some(budget) = config.thinking_budget_tokens {
        max_tokens = max_tokens.max(budget.saturating_add(1));
    }

    let messages_json: Vec<Value> = wire_messages
        .into_iter()
        .map(|wm| {
            json!({
                "role": wm.role,
                "content": wm.blocks,
            })
        })
        .collect();

    let mut payload = json!({
        "model": req.model,
        "max_tokens": max_tokens,
        "stream": true,
        "messages": messages_json,
    });

    if !system_blocks.is_empty() {
        payload["system"] = Value::Array(system_blocks);
    }
    if !wire_tools.is_empty() {
        payload["tools"] = Value::Array(wire_tools);
    }
    if let Some(budget) = config.thinking_budget_tokens {
        payload["thinking"] = json!({
            "type": "enabled",
            "budget_tokens": budget,
        });
    } else if let Some(temp) = req.temperature {
        payload["temperature"] = json!(temp);
    }

    payload
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorEnvelope {
    error: AnthropicErrorBody,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorBody {
    #[serde(rename = "type", default)]
    error_type: String,
    #[serde(default)]
    message: String,
}

/// Map non-2xx Anthropic HTTP status + JSON body into a structured [`AiError`].
pub fn classify_http_error(status: u16, body: &str, retry_after_ms: Option<u64>) -> AiError {
    let parsed = serde_json::from_str::<AnthropicErrorEnvelope>(body).ok();
    let error_type = parsed
        .as_ref()
        .map_or("", |env| env.error.error_type.as_str());
    let detail = parsed
        .as_ref()
        .map(|env| env.error.message.as_str())
        .filter(|msg| !msg.is_empty())
        .unwrap_or(body);

    classify_error_parts(Some(status), error_type, detail, retry_after_ms)
}

fn classify_stream_error(payload: &str) -> AiError {
    let parsed = serde_json::from_str::<AnthropicErrorEnvelope>(payload).ok();
    let error_type = parsed
        .as_ref()
        .map_or("", |env| env.error.error_type.as_str());
    let detail = parsed
        .as_ref()
        .map(|env| env.error.message.as_str())
        .filter(|msg| !msg.is_empty())
        .unwrap_or(payload);

    classify_error_parts(None, error_type, detail, None)
}

fn classify_error_parts(
    status: Option<u16>,
    error_type: &str,
    message: &str,
    retry_after_ms: Option<u64>,
) -> AiError {
    let lower_type = error_type.to_ascii_lowercase();
    let lower_msg = message.to_ascii_lowercase();

    if matches!(status, Some(401 | 403))
        || matches!(
            lower_type.as_str(),
            "authentication_error" | "permission_error"
        )
    {
        return AiError::Authentication(match status {
            Some(code) => format!("HTTP {code}: {message}"),
            None => message.to_string(),
        });
    }

    if status == Some(429) || lower_type == "rate_limit_error" {
        return AiError::RateLimited {
            retry_after_ms,
            message: message.to_string(),
        };
    }

    if status == Some(529) || lower_type == "overloaded_error" {
        return AiError::Http {
            status: status.unwrap_or(529),
            message: message.to_string(),
            retry_after_ms,
        };
    }

    if matches!(status, None | Some(400 | 413)) && is_context_overflow_message(&lower_msg) {
        return AiError::ContextOverflow {
            message: message.to_string(),
        };
    }

    if status == Some(400) || lower_type == "invalid_request_error" {
        return AiError::InvalidRequest(message.to_string());
    }

    match status {
        Some(code) => AiError::Http {
            status: code,
            message: message.to_string(),
            retry_after_ms,
        },
        None => AiError::Protocol(format!("anthropic stream error ({error_type}): {message}")),
    }
}

fn is_context_overflow_message(lower: &str) -> bool {
    lower.contains("prompt is too long")
        || lower.contains("too many tokens")
        || lower.contains("context length")
        || lower.contains("maximum context")
        || lower.contains("exceed context limit")
}

/// Result of feeding one line into [`AnthropicSseParser`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseParseStep {
    Continue(Vec<StreamEvent>),
    Stop(Vec<StreamEvent>),
}

#[derive(Debug, Default)]
struct UsageAccumulator {
    uncached_input: u64,
    cache_write: u64,
    cache_read: u64,
    output: u64,
}

impl UsageAccumulator {
    fn merge_start(&mut self, usage: &StartUsage) {
        self.uncached_input = usage.input;
        self.cache_write = usage.cache_creation_input;
        self.cache_read = usage.cache_read_input;
        self.output = usage.output;
    }

    fn merge_delta(&mut self, usage: &DeltaUsage) {
        if let Some(v) = usage.input {
            self.uncached_input = v;
        }
        if let Some(v) = usage.cache_creation_input {
            self.cache_write = v;
        }
        if let Some(v) = usage.cache_read_input {
            self.cache_read = v;
        }
        if let Some(v) = usage.output {
            self.output = v;
        }
    }

    fn to_token_usage(&self) -> TokenUsage {
        let total_input = self
            .uncached_input
            .saturating_add(self.cache_write)
            .saturating_add(self.cache_read);
        TokenUsage {
            input_tokens: total_input,
            output_tokens: self.output,
            cache_read_tokens: self.cache_read,
            cache_write_tokens: self.cache_write,
            reasoning_tokens: 0,
        }
    }
}

/// Stateful parser for Anthropic's named-event SSE protocol (`event:` + `data:` + blank line).
#[derive(Debug, Default)]
pub struct AnthropicSseParser {
    current_event: Option<String>,
    data_lines: Vec<String>,
    pending_signatures: BTreeMap<usize, String>,
    usage: UsageAccumulator,
    has_usage: bool,
    stop_reason: Option<FinishReason>,
    finished: bool,
}

impl AnthropicSseParser {
    /// Feed one decoded line (without trailing `\r`/`\n`) into the SSE state machine.
    pub fn push_line(&mut self, line: &str) -> Result<SseParseStep, AiError> {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            return self.dispatch_buffered_event();
        }
        if line.starts_with(':') {
            return Ok(SseParseStep::Continue(Vec::new()));
        }
        if let Some(event_name) = line.strip_prefix("event:") {
            self.current_event = Some(event_name.trim().to_string());
            return Ok(SseParseStep::Continue(Vec::new()));
        }
        if let Some(data_rest) = line.strip_prefix("data:") {
            let trimmed = data_rest.strip_prefix(' ').unwrap_or(data_rest);
            self.data_lines.push(trimmed.to_string());
        }
        Ok(SseParseStep::Continue(Vec::new()))
    }

    /// Flush any pending buffered event when the underlying byte stream hits EOF.
    pub fn finish_eof(&mut self) -> Result<Vec<StreamEvent>, AiError> {
        let mut out = Vec::new();
        if !self.data_lines.is_empty() || self.current_event.is_some() {
            match self.dispatch_buffered_event()? {
                SseParseStep::Continue(evs) | SseParseStep::Stop(evs) => out.extend(evs),
            }
        }
        if !self.finished && (self.has_usage || self.stop_reason.is_some()) {
            self.finalize_events(&mut out);
        }
        Ok(out)
    }

    fn dispatch_buffered_event(&mut self) -> Result<SseParseStep, AiError> {
        let event_name = self.current_event.take();
        if self.data_lines.is_empty() {
            return Ok(SseParseStep::Continue(Vec::new()));
        }
        let payload = self.data_lines.join("\n");
        self.data_lines.clear();

        let kind = match event_name.as_deref() {
            Some(name) if !name.is_empty() => name.to_string(),
            _ => {
                let envelope: GenericEventType = serde_json::from_str(&payload)
                    .map_err(|e| AiError::Protocol(format!("invalid Anthropic SSE JSON: {e}")))?;
                envelope.event_type.unwrap_or_default()
            }
        };

        match kind.as_str() {
            "error" => Err(classify_stream_error(&payload)),
            "message_start" => self.handle_message_start(&payload),
            "content_block_start" => self.handle_block_start(&payload),
            "content_block_delta" => self.handle_block_delta(&payload),
            "content_block_stop" => self.handle_block_stop(&payload),
            "message_delta" => self.handle_message_delta(&payload),
            "message_stop" => {
                let mut events = Vec::new();
                self.finalize_events(&mut events);
                Ok(SseParseStep::Stop(events))
            }
            _ => Ok(SseParseStep::Continue(Vec::new())),
        }
    }

    fn handle_message_start(&mut self, payload: &str) -> Result<SseParseStep, AiError> {
        let ev: MessageStartPayload = serde_json::from_str(payload)
            .map_err(|e| AiError::Protocol(format!("invalid message_start payload: {e}")))?;
        if let Some(usage) = ev.message.and_then(|m| m.usage) {
            self.usage.merge_start(&usage);
            self.has_usage = true;
        }
        Ok(SseParseStep::Continue(Vec::new()))
    }

    fn handle_block_start(&mut self, payload: &str) -> Result<SseParseStep, AiError> {
        let ev: ContentBlockStartPayload = serde_json::from_str(payload)
            .map_err(|e| AiError::Protocol(format!("invalid content_block_start payload: {e}")))?;
        let mut events = Vec::new();
        match ev.content_block {
            WireContentBlock::Text { text } if !text.is_empty() => {
                events.push(StreamEvent::TextDelta { text });
            }
            WireContentBlock::Thinking {
                thinking,
                signature,
            } => {
                if !thinking.is_empty() {
                    events.push(StreamEvent::ThinkingDelta { text: thinking });
                }
                if !signature.is_empty() {
                    self.pending_signatures
                        .entry(ev.index)
                        .or_default()
                        .push_str(&signature);
                }
            }
            WireContentBlock::RedactedThinking { data } if !data.is_empty() => {
                events.push(StreamEvent::ThinkingSignature { signature: data });
            }
            WireContentBlock::ToolUse { id, name, input } => {
                events.push(StreamEvent::ToolCallStart {
                    index: ev.index,
                    id,
                    name,
                });
                if let Some(Value::Object(map)) = input
                    && !map.is_empty()
                    && let Ok(fragment) = serde_json::to_string(&Value::Object(map))
                {
                    events.push(StreamEvent::ToolCallArgsDelta {
                        index: ev.index,
                        json_fragment: fragment,
                    });
                }
            }
            _ => {}
        }
        Ok(SseParseStep::Continue(events))
    }

    fn handle_block_delta(&mut self, payload: &str) -> Result<SseParseStep, AiError> {
        let ev: ContentBlockDeltaPayload = serde_json::from_str(payload)
            .map_err(|e| AiError::Protocol(format!("invalid content_block_delta payload: {e}")))?;
        let mut events = Vec::new();
        match ev.delta {
            WireBlockDelta::TextDelta { text } if !text.is_empty() => {
                events.push(StreamEvent::TextDelta { text });
            }
            WireBlockDelta::ThinkingDelta { thinking } if !thinking.is_empty() => {
                events.push(StreamEvent::ThinkingDelta { text: thinking });
            }
            WireBlockDelta::SignatureDelta { signature } if !signature.is_empty() => {
                self.pending_signatures
                    .entry(ev.index)
                    .or_default()
                    .push_str(&signature);
            }
            WireBlockDelta::InputJsonDelta { partial_json } if !partial_json.is_empty() => {
                events.push(StreamEvent::ToolCallArgsDelta {
                    index: ev.index,
                    json_fragment: partial_json,
                });
            }
            _ => {}
        }
        Ok(SseParseStep::Continue(events))
    }

    fn handle_block_stop(&mut self, payload: &str) -> Result<SseParseStep, AiError> {
        let ev: ContentBlockStopPayload = serde_json::from_str(payload)
            .map_err(|e| AiError::Protocol(format!("invalid content_block_stop payload: {e}")))?;
        let mut events = Vec::new();
        if let Some(sig) = self.pending_signatures.remove(&ev.index)
            && !sig.is_empty()
        {
            events.push(StreamEvent::ThinkingSignature { signature: sig });
        }
        Ok(SseParseStep::Continue(events))
    }

    fn handle_message_delta(&mut self, payload: &str) -> Result<SseParseStep, AiError> {
        let ev: MessageDeltaPayload = serde_json::from_str(payload)
            .map_err(|e| AiError::Protocol(format!("invalid message_delta payload: {e}")))?;
        if let Some(usage) = ev.usage {
            self.usage.merge_delta(&usage);
            self.has_usage = true;
        }
        if let Some(reason) = ev.delta.and_then(|d| d.stop_reason) {
            self.stop_reason = Some(map_stop_reason(&reason));
        }
        Ok(SseParseStep::Continue(Vec::new()))
    }

    fn finalize_events(&mut self, out: &mut Vec<StreamEvent>) {
        if self.finished {
            return;
        }
        self.finished = true;
        let leftover = std::mem::take(&mut self.pending_signatures);
        for (_, sig) in leftover {
            if !sig.is_empty() {
                out.push(StreamEvent::ThinkingSignature { signature: sig });
            }
        }
        if self.has_usage {
            out.push(StreamEvent::Usage(self.usage.to_token_usage()));
        }
        out.push(StreamEvent::Done {
            reason: self.stop_reason.take().unwrap_or(FinishReason::Stop),
        });
    }
}

fn map_stop_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "tool_use" => FinishReason::ToolCalls,
        "max_tokens" | "model_context_window_exceeded" => FinishReason::Length,
        "refusal" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_string()),
    }
}

#[derive(Debug, Deserialize)]
struct GenericEventType {
    #[serde(rename = "type", default)]
    event_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageStartPayload {
    #[serde(default)]
    message: Option<StartMessage>,
}

#[derive(Debug, Deserialize)]
struct StartMessage {
    #[serde(default)]
    usage: Option<StartUsage>,
}

#[derive(Debug, Deserialize)]
struct StartUsage {
    #[serde(rename = "input_tokens", default)]
    input: u64,
    #[serde(rename = "cache_creation_input_tokens", default)]
    cache_creation_input: u64,
    #[serde(rename = "cache_read_input_tokens", default)]
    cache_read_input: u64,
    #[serde(rename = "output_tokens", default)]
    output: u64,
}

#[derive(Debug, Deserialize)]
struct ContentBlockStartPayload {
    index: usize,
    content_block: WireContentBlock,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireContentBlock {
    Text {
        #[serde(default)]
        text: String,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: String,
    },
    RedactedThinking {
        #[serde(default)]
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Option<Value>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct ContentBlockDeltaPayload {
    index: usize,
    delta: WireBlockDelta,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireBlockDelta {
    TextDelta {
        #[serde(default)]
        text: String,
    },
    ThinkingDelta {
        #[serde(default)]
        thinking: String,
    },
    SignatureDelta {
        #[serde(default)]
        signature: String,
    },
    InputJsonDelta {
        #[serde(default)]
        partial_json: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct ContentBlockStopPayload {
    index: usize,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaPayload {
    #[serde(default)]
    delta: Option<MessageDeltaBody>,
    #[serde(default)]
    usage: Option<DeltaUsage>,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaBody {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeltaUsage {
    #[serde(rename = "input_tokens", default)]
    input: Option<u64>,
    #[serde(rename = "cache_creation_input_tokens", default)]
    cache_creation_input: Option<u64>,
    #[serde(rename = "cache_read_input_tokens", default)]
    cache_read_input: Option<u64>,
    #[serde(rename = "output_tokens", default)]
    output: Option<u64>,
}

fn build_anthropic_event_stream<S, B, E>(byte_stream: S, cancel: CancellationToken) -> EventStream
where
    S: futures::Stream<Item = Result<B, E>> + Send + Unpin + 'static,
    B: AsRef<[u8]> + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    let state = (
        byte_stream,
        SseLineDecoder::default(),
        AnthropicSseParser::default(),
        VecDeque::<Result<StreamEvent, AiError>>::new(),
        cancel,
        false,
    );

    let stream = futures::stream::unfold(
        state,
        |(mut byte_stream, mut decoder, mut parser, mut pending, cancel, mut done)| async move {
            loop {
                if let Some(item) = pending.pop_front() {
                    return Some((item, (byte_stream, decoder, parser, pending, cancel, done)));
                }
                if done {
                    return None;
                }

                if cancel.is_cancelled() {
                    return Some((
                        Err(AiError::Cancelled),
                        (byte_stream, decoder, parser, pending, cancel, true),
                    ));
                }

                let next_chunk = tokio::select! {
                    biased;
                    () = cancel.cancelled() => {
                        return Some((
                            Err(AiError::Cancelled),
                            (byte_stream, decoder, parser, pending, cancel, true),
                        ));
                    }
                    chunk = byte_stream.next() => chunk,
                };

                match next_chunk {
                    Some(Ok(bytes)) => {
                        for line in decoder.push_bytes(bytes.as_ref()) {
                            match parser.push_line(&line) {
                                Ok(SseParseStep::Continue(events)) => {
                                    for ev in events {
                                        pending.push_back(Ok(ev));
                                    }
                                }
                                Ok(SseParseStep::Stop(events)) => {
                                    for ev in events {
                                        pending.push_back(Ok(ev));
                                    }
                                    done = true;
                                    break;
                                }
                                Err(err) => {
                                    pending.push_back(Err(err));
                                    done = true;
                                    break;
                                }
                            }
                        }
                    }
                    Some(Err(err)) => {
                        return Some((
                            Err(AiError::Network(err.to_string())),
                            (byte_stream, decoder, parser, pending, cancel, true),
                        ));
                    }
                    None => {
                        if let Some(rem) = decoder.flush_remaining()
                            && let Err(err) = push_line_into_queue(&mut parser, &rem, &mut pending)
                        {
                            pending.push_back(Err(err));
                        } else {
                            match parser.finish_eof() {
                                Ok(events) => {
                                    for ev in events {
                                        pending.push_back(Ok(ev));
                                    }
                                }
                                Err(err) => pending.push_back(Err(err)),
                            }
                        }
                        done = true;
                    }
                }
            }
        },
    );

    Box::pin(stream)
}

fn push_line_into_queue(
    parser: &mut AnthropicSseParser,
    line: &str,
    pending: &mut VecDeque<Result<StreamEvent, AiError>>,
) -> Result<(), AiError> {
    match parser.push_line(line)? {
        SseParseStep::Continue(events) | SseParseStep::Stop(events) => {
            for ev in events {
                pending.push_back(Ok(ev));
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ErrorCategory, ToolDefinition};

    fn count_cache_control(val: &Value) -> usize {
        match val {
            Value::Array(items) => items.iter().map(count_cache_control).sum(),
            Value::Object(map) => {
                usize::from(map.contains_key("cache_control"))
                    + map.values().map(count_cache_control).sum::<usize>()
            }
            _ => 0,
        }
    }

    #[test]
    fn build_request_body_places_four_cache_breakpoints_and_merges_tool_results() {
        let req = ProviderRequest {
            model: "claude-sonnet-4-6".into(),
            system: Some("You are OrangeAgent.".into()),
            messages: vec![
                Message::user("inspect Cargo.toml and src/lib.rs"),
                Message {
                    role: Role::Assistant,
                    content: vec![
                        ContentBlock::Thinking {
                            text: "let me read both".into(),
                            signature: Some("anth_sig_valid".into()),
                            provider: Some(ProviderFamily::Anthropic),
                        },
                        ContentBlock::Thinking {
                            text: "foreign reasoning".into(),
                            signature: Some("openai_enc_sig".into()),
                            provider: Some(ProviderFamily::OpenAiResponses),
                        },
                        ContentBlock::Text {
                            text: "Reading both files.".into(),
                        },
                        ContentBlock::ToolCall {
                            id: "toolu_1".into(),
                            name: "read".into(),
                            arguments: json!({ "path": "Cargo.toml" }),
                        },
                        ContentBlock::ToolCall {
                            id: "toolu_2".into(),
                            name: "read".into(),
                            arguments: json!({ "path": "src/lib.rs" }),
                        },
                    ],
                },
                Message::tool_result("toolu_1", "[package]\nname = \"demo\"", false),
                Message::tool_result("toolu_2", "missing file", true),
                Message::user("also check README.md"),
            ],
            tools: vec![
                ToolDefinition {
                    name: "grep".into(),
                    description: "Search files".into(),
                    parameters: json!({ "type": "object" }),
                },
                ToolDefinition {
                    name: "read".into(),
                    description: "Read a file".into(),
                    parameters: json!({ "type": "object", "required": ["path"] }),
                },
            ],
            max_tokens: Some(4_096),
            temperature: Some(0.2),
        };

        let config = AnthropicConfig::new("sk-ant-test");
        let body = build_request_body(&req, &config);

        // Exactly 4 explicit cache_control breakpoints:
        // 1) last tool, 2) system block, 3) assistant turn, 4) merged user/tool_result turn
        assert_eq!(count_cache_control(&body), 4);
        assert!(body["tools"][0].get("cache_control").is_none());
        assert_eq!(
            body["tools"][1]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert!(body["tools"][1].get("input_schema").is_some());
        assert_eq!(
            body["system"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );

        let msgs = body["messages"].as_array().expect("messages array");
        // Merged into 3 alternating turns: User -> Assistant -> User(tool_result, tool_result, text)
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert!(msgs[0]["content"][0].get("cache_control").is_none());

        assert_eq!(msgs[1]["role"], "assistant");
        let asst_blocks = msgs[1]["content"].as_array().expect("assistant blocks");
        // Foreign thinking stripped & unsigned thinking dropped; remaining:
        // 0: valid Anthropic thinking, 1: text, 2: tool_use_1, 3: tool_use_2 (with cache_control)
        assert_eq!(asst_blocks.len(), 4);
        assert_eq!(asst_blocks[0]["type"], "thinking");
        assert_eq!(asst_blocks[0]["signature"], "anth_sig_valid");
        assert!(asst_blocks[0].get("cache_control").is_none());
        assert_eq!(asst_blocks[2]["type"], "tool_use");
        assert_eq!(asst_blocks[2]["input"], json!({ "path": "Cargo.toml" }));
        assert_eq!(
            asst_blocks[3]["cache_control"],
            json!({ "type": "ephemeral" })
        );

        assert_eq!(msgs[2]["role"], "user");
        let user_blocks = msgs[2]["content"].as_array().expect("merged user blocks");
        assert_eq!(user_blocks.len(), 3);
        assert_eq!(user_blocks[0]["type"], "tool_result");
        assert_eq!(user_blocks[0]["tool_use_id"], "toolu_1");
        assert!(user_blocks[0].get("is_error").is_none());
        assert_eq!(user_blocks[1]["type"], "tool_result");
        assert_eq!(user_blocks[1]["is_error"], json!(true));
        assert_eq!(user_blocks[2]["type"], "text");
        assert_eq!(
            user_blocks[2]["cache_control"],
            json!({ "type": "ephemeral" })
        );
    }

    #[test]
    fn build_request_body_respects_cache_disabled_and_thinking_budget() {
        let req = ProviderRequest {
            model: "claude-opus-4-6".into(),
            system: Some("System prompt".into()),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: Some(2_048),
            temperature: Some(0.7),
        };

        let disabled = AnthropicConfig::new("sk-ant").with_cache_enabled(false);
        let body_disabled = build_request_body(&req, &disabled);
        assert_eq!(count_cache_control(&body_disabled), 0);
        assert!((body_disabled["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-6);

        let thinking_cfg = AnthropicConfig::new("sk-ant")
            .with_cache_ttl(CacheTtl::Ephemeral1h)
            .with_thinking_budget(Some(4_096));
        let body_thinking = build_request_body(&req, &thinking_cfg);
        assert_eq!(
            body_thinking["thinking"],
            json!({ "type": "enabled", "budget_tokens": 4_096 })
        );
        // max_tokens automatically raised above thinking budget
        assert!(body_thinking["max_tokens"].as_u64().unwrap() > 4_096);
        assert!(body_thinking.get("temperature").is_none());
        assert_eq!(
            body_thinking["system"][0]["cache_control"],
            json!({ "type": "ephemeral", "ttl": "1h" })
        );
    }

    #[tokio::test]
    async fn anthropic_sse_stream_parses_thinking_signature_tool_use_and_merged_usage() {
        let raw_sse = concat!(
            ": keepalive\n\n",
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":100,\"cache_creation_input_tokens\":400,\"cache_read_input_tokens\":1500,\"output_tokens\":1}}}\n\n",
            "event: ping\n",
            "data: {\"type\":\"ping\"}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"先读\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_part1_\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"part2\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"你好\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_9\",\"name\":\"read\",\"input\":{}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"Cargo.toml\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":2}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":60}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n"
        );

        // Split arbitrarily at odd byte offsets (including inside UTF-8 multi-byte characters)
        let bytes = raw_sse.as_bytes();
        let mut chunks: Vec<Result<Vec<u8>, std::io::Error>> = Vec::new();
        let mut offset = 0;
        let step = 17;
        while offset < bytes.len() {
            let end = (offset + step).min(bytes.len());
            chunks.push(Ok(bytes[offset..end].to_vec()));
            offset = end;
        }

        let stream =
            build_anthropic_event_stream(futures::stream::iter(chunks), CancellationToken::new());
        let events: Vec<StreamEvent> = stream
            .map(|item| item.expect("valid stream event"))
            .collect()
            .await;

        assert_eq!(
            events,
            vec![
                StreamEvent::ThinkingDelta {
                    text: "先读".into()
                },
                StreamEvent::ThinkingSignature {
                    signature: "sig_part1_part2".into()
                },
                StreamEvent::TextDelta {
                    text: "你好".into()
                },
                StreamEvent::ToolCallStart {
                    index: 2,
                    id: "toolu_9".into(),
                    name: "read".into(),
                },
                StreamEvent::ToolCallArgsDelta {
                    index: 2,
                    json_fragment: "{\"path\":".into(),
                },
                StreamEvent::ToolCallArgsDelta {
                    index: 2,
                    json_fragment: "\"Cargo.toml\"}".into(),
                },
                StreamEvent::Usage(TokenUsage {
                    input_tokens: 2_000,
                    output_tokens: 60,
                    cache_read_tokens: 1_500,
                    cache_write_tokens: 400,
                    reasoning_tokens: 0,
                }),
                StreamEvent::Done {
                    reason: FinishReason::ToolCalls,
                },
            ]
        );

        if let Some(StreamEvent::Usage(u)) =
            events.iter().find(|e| matches!(e, StreamEvent::Usage(_)))
        {
            assert!((u.cache_hit_ratio() - 0.75).abs() < 1e-9);
        } else {
            panic!("missing usage event");
        }
    }

    #[tokio::test]
    async fn anthropic_sse_stream_maps_in_stream_error_and_cancellation() {
        let err_sse = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_2\"}}\n\n",
            "event: error\n",
            "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
        );

        let err_stream = build_anthropic_event_stream(
            futures::stream::iter(vec![Ok::<_, std::io::Error>(err_sse.as_bytes().to_vec())]),
            CancellationToken::new(),
        );
        let items: Vec<Result<StreamEvent, AiError>> = err_stream.collect().await;
        assert_eq!(items.len(), 1);
        let err = items[0].as_ref().expect_err("expected overloaded error");
        let verdict = err.verdict();
        assert_eq!(verdict.category, ErrorCategory::ServerError);
        assert!(verdict.retryable);

        let cancel = CancellationToken::new();
        cancel.cancel();
        let cancelled_stream = build_anthropic_event_stream(
            futures::stream::iter(vec![Ok::<_, std::io::Error>(b"event: ping\n\n".to_vec())]),
            cancel,
        );
        let cancelled_items: Vec<Result<StreamEvent, AiError>> = cancelled_stream.collect().await;
        assert_eq!(cancelled_items.len(), 1);
        assert!(matches!(cancelled_items[0], Err(AiError::Cancelled)));
    }

    #[test]
    fn classify_http_error_handles_anthropic_overflow_rate_limit_and_overloaded() {
        let overflow = classify_http_error(
            400,
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens > 200000 maximum"}}"#,
            None,
        );
        let overflow_verdict = overflow.verdict();
        assert_eq!(overflow_verdict.category, ErrorCategory::ContextOverflow);
        assert!(overflow_verdict.should_compress);

        let rate = classify_http_error(
            429,
            r#"{"type":"error","error":{"type":"rate_limit_error","message":"Rate limit exceeded"}}"#,
            Some(3_000),
        );
        assert_eq!(rate.verdict().category, ErrorCategory::RateLimit);
        assert_eq!(rate.verdict().retry_after_ms, Some(3_000));

        let overloaded = classify_http_error(
            529,
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            None,
        );
        assert_eq!(overloaded.verdict().category, ErrorCategory::ServerError);
        assert!(overloaded.verdict().retryable);

        let auth = classify_http_error(
            401,
            r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#,
            None,
        );
        assert_eq!(auth.verdict().category, ErrorCategory::Auth);
        assert!(!auth.verdict().retryable);
    }
}
