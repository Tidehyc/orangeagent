use std::collections::VecDeque;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::{
    AiError, ContentBlock, EventStream, FinishReason, LlmProvider, Message, ProviderFamily,
    ProviderRequest, Role, StreamEvent, TokenUsage, sanitize_for_provider,
};

/// Configuration for an OpenAI Chat Completions compatible provider
/// (works with OpenAI, DeepSeek, OpenRouter, Ollama, vLLM, etc.).
#[derive(Debug, Clone)]
pub struct OpenAiChatConfig {
    /// Base URL without trailing `/chat/completions`, e.g. `https://api.deepseek.com/v1`.
    pub base_url: String,
    /// Bearer token (optional for local runtimes such as Ollama/vLLM).
    pub api_key: Option<String>,
}

impl OpenAiChatConfig {
    pub fn new(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key,
        }
    }

    fn endpoint_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        format!("{base}/chat/completions")
    }
}

/// Streaming client for OpenAI Chat Completions compatible APIs.
#[derive(Debug, Clone)]
pub struct OpenAiChatProvider {
    config: OpenAiChatConfig,
    client: reqwest::Client,
}

impl OpenAiChatProvider {
    pub fn new(config: OpenAiChatConfig) -> Result<Self, AiError> {
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| AiError::Network(format!("failed to build HTTP client: {e}")))?;
        Ok(Self { config, client })
    }
}

#[async_trait]
impl LlmProvider for OpenAiChatProvider {
    fn family(&self) -> ProviderFamily {
        ProviderFamily::OpenAiChat
    }

    async fn stream(
        &self,
        req: ProviderRequest,
        cancel: CancellationToken,
    ) -> Result<EventStream, AiError> {
        let body = build_request_body(&req);
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(key) = &self.config.api_key {
            let value = HeaderValue::from_str(&format!("Bearer {key}"))
                .map_err(|e| AiError::Authentication(format!("invalid API key header: {e}")))?;
            headers.insert(AUTHORIZATION, value);
        }

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
        Ok(build_sse_event_stream(byte_stream, cancel))
    }
}

/// Convert a normalized [`ProviderRequest`] into the OpenAI Chat Completions JSON payload.
pub fn build_request_body(req: &ProviderRequest) -> Value {
    let sanitized = sanitize_for_provider(&req.messages, ProviderFamily::OpenAiChat);
    let mut wire_messages = Vec::with_capacity(sanitized.len() + usize::from(req.system.is_some()));

    if let Some(system) = &req.system {
        wire_messages.push(json!({
            "role": "system",
            "content": system,
        }));
    }

    for msg in &sanitized {
        push_wire_messages(&mut wire_messages, msg);
    }

    let mut payload = json!({
        "model": req.model,
        "messages": wire_messages,
        "stream": true,
        "stream_options": {
            "include_usage": true
        }
    });

    if !req.tools.is_empty() {
        let wire_tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect();
        payload["tools"] = Value::Array(wire_tools);
    }

    if let Some(max_tokens) = req.max_tokens {
        payload["max_tokens"] = json!(max_tokens);
    }
    if let Some(temperature) = req.temperature {
        payload["temperature"] = json!(temperature);
    }

    payload
}

fn push_wire_messages(out: &mut Vec<Value>, msg: &Message) {
    match msg.role {
        Role::System => {
            out.push(json!({
                "role": "system",
                "content": msg.text_content(),
            }));
        }
        Role::User => {
            out.push(json!({
                "role": "user",
                "content": msg.text_content(),
            }));
        }
        Role::Assistant => {
            let mut text = String::new();
            let mut reasoning = String::new();
            let mut tool_calls = Vec::new();

            for block in &msg.content {
                match block {
                    ContentBlock::Text { text: t } => text.push_str(t),
                    ContentBlock::Thinking { text: t, .. } => reasoning.push_str(t),
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } => {
                        let args_str = serde_json::to_string(arguments).unwrap_or_default();
                        tool_calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args_str,
                            }
                        }));
                    }
                    ContentBlock::ToolResult { .. } => {}
                }
            }

            let mut obj = serde_json::Map::new();
            obj.insert("role".into(), json!("assistant"));
            if !text.is_empty() || tool_calls.is_empty() {
                obj.insert("content".into(), json!(text));
            }
            if !reasoning.is_empty() {
                obj.insert("reasoning_content".into(), json!(reasoning));
            }
            if !tool_calls.is_empty() {
                obj.insert("tool_calls".into(), Value::Array(tool_calls));
            }
            out.push(Value::Object(obj));
        }
        Role::Tool => {
            for block in &msg.content {
                if let ContentBlock::ToolResult {
                    tool_call_id,
                    content,
                    ..
                } = block
                {
                    out.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,
                    }));
                }
            }
        }
    }
}

/// Map non-2xx HTTP status + response body into a structured [`AiError`].
pub fn classify_http_error(status: u16, body: &str, retry_after_ms: Option<u64>) -> AiError {
    let lower = body.to_ascii_lowercase();
    if status == 401 || status == 403 {
        return AiError::Authentication(format!("HTTP {status}: {body}"));
    }
    if status == 429 {
        return AiError::RateLimited {
            retry_after_ms,
            message: body.to_string(),
        };
    }
    if (status == 400 || status == 413)
        && (lower.contains("context_length_exceeded")
            || lower.contains("maximum context length")
            || lower.contains("too many tokens"))
    {
        return AiError::ContextOverflow {
            message: body.to_string(),
        };
    }
    AiError::Http {
        status,
        message: body.to_string(),
        retry_after_ms,
    }
}

/// Incremental UTF-8-safe line splitter for Server-Sent Events (SSE) byte streams.
///
/// Buffers raw bytes until `\n` so multi-byte UTF-8 characters split across TCP
/// packets are never corrupted.
#[derive(Debug, Default)]
pub struct SseLineDecoder {
    buf: Vec<u8>,
}

impl SseLineDecoder {
    pub fn push_bytes(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut lines = Vec::new();
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let mut line_bytes: Vec<u8> = self.buf.drain(..=pos).collect();
            // Drop trailing '\n' and optional '\r'
            line_bytes.pop();
            if line_bytes.last() == Some(&b'\r') {
                line_bytes.pop();
            }
            lines.push(String::from_utf8_lossy(&line_bytes).into_owned());
        }
        lines
    }

    pub fn flush_remaining(&mut self) -> Option<String> {
        if self.buf.is_empty() {
            return None;
        }
        let mut bytes = std::mem::take(&mut self.buf);
        if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
        Some(String::from_utf8_lossy(&bytes).into_owned())
    }
}

/// Parse one SSE line (e.g. `data: {...}` or `data: [DONE]`) into zero or more [`StreamEvent`]s.
///
/// Returns `Ok(None)` when the terminal `[DONE]` sentinel is encountered.
pub fn parse_sse_line(line: &str) -> Result<Option<Vec<StreamEvent>>, AiError> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with(':') {
        return Ok(Some(Vec::new()));
    }
    let Some(payload) = trimmed.strip_prefix("data:") else {
        return Ok(Some(Vec::new()));
    };
    let payload = payload.trim();
    if payload == "[DONE]" {
        return Ok(None);
    }

    let chunk: ChatCompletionChunk = serde_json::from_str(payload).map_err(|e| {
        AiError::Protocol(format!("invalid SSE JSON chunk: {e} (payload={payload})"))
    })?;
    Ok(Some(convert_chunk(chunk)))
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: ChunkDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ChunkDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<WireToolCallDelta>,
}

#[derive(Debug, Deserialize)]
struct WireToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<WireFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct WireFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
    /// DeepSeek compatibility field for cached prompt tokens.
    #[serde(default)]
    prompt_cache_hit_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

fn convert_chunk(chunk: ChatCompletionChunk) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    for choice in chunk.choices {
        if let Some(reasoning) = choice.delta.reasoning_content
            && !reasoning.is_empty()
        {
            events.push(StreamEvent::ThinkingDelta { text: reasoning });
        }

        if let Some(content) = choice.delta.content
            && !content.is_empty()
        {
            events.push(StreamEvent::TextDelta { text: content });
        }

        for tc in choice.delta.tool_calls {
            let fn_name = tc.function.as_ref().and_then(|f| f.name.clone());
            if let (Some(id), Some(name)) = (tc.id, fn_name) {
                events.push(StreamEvent::ToolCallStart {
                    index: tc.index,
                    id,
                    name,
                });
            }
            if let Some(args) = tc.function.and_then(|f| f.arguments)
                && !args.is_empty()
            {
                events.push(StreamEvent::ToolCallArgsDelta {
                    index: tc.index,
                    json_fragment: args,
                });
            }
        }

        if let Some(reason) = choice.finish_reason {
            let mapped = match reason.as_str() {
                "stop" => FinishReason::Stop,
                "tool_calls" => FinishReason::ToolCalls,
                "length" => FinishReason::Length,
                "content_filter" => FinishReason::ContentFilter,
                other => FinishReason::Other(other.to_string()),
            };
            events.push(StreamEvent::Done { reason: mapped });
        }
    }

    if let Some(usage) = chunk.usage {
        let cache_read = usage
            .prompt_tokens_details
            .map(|d| d.cached_tokens)
            .or(usage.prompt_cache_hit_tokens)
            .unwrap_or(0);
        let reasoning = usage
            .completion_tokens_details
            .map_or(0, |d| d.reasoning_tokens);
        events.push(StreamEvent::Usage(TokenUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            cache_read_tokens: cache_read,
            cache_write_tokens: 0,
            reasoning_tokens: reasoning,
        }));
    }

    events
}

fn build_sse_event_stream<S, B, E>(byte_stream: S, cancel: CancellationToken) -> EventStream
where
    S: futures::Stream<Item = Result<B, E>> + Send + Unpin + 'static,
    B: AsRef<[u8]> + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    let state = (
        byte_stream,
        SseLineDecoder::default(),
        VecDeque::<Result<StreamEvent, AiError>>::new(),
        cancel,
        false,
    );

    let stream = futures::stream::unfold(
        state,
        |(mut byte_stream, mut decoder, mut pending, cancel, mut done)| async move {
            loop {
                if let Some(item) = pending.pop_front() {
                    return Some((item, (byte_stream, decoder, pending, cancel, done)));
                }
                if done {
                    return None;
                }

                let next_chunk = tokio::select! {
                    () = cancel.cancelled() => {
                        return Some((
                            Err(AiError::Cancelled),
                            (byte_stream, decoder, pending, cancel, true),
                        ));
                    }
                    chunk = byte_stream.next() => chunk,
                };

                match next_chunk {
                    Some(Ok(bytes)) => {
                        for line in decoder.push_bytes(bytes.as_ref()) {
                            match parse_sse_line(&line) {
                                Ok(Some(events)) => {
                                    for ev in events {
                                        pending.push_back(Ok(ev));
                                    }
                                }
                                Ok(None) => {
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
                            (byte_stream, decoder, pending, cancel, true),
                        ));
                    }
                    None => {
                        if let Some(rem) = decoder.flush_remaining() {
                            match parse_sse_line(&rem) {
                                Ok(Some(events)) => {
                                    for ev in events {
                                        pending.push_back(Ok(ev));
                                    }
                                }
                                Ok(None) => {}
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ErrorCategory, ToolDefinition};

    #[test]
    fn sse_decoder_handles_split_tcp_chunks_and_utf8_boundaries() {
        let mut decoder = SseLineDecoder::default();
        let raw = "data: {\"hello\":\"你好Rust\"}\r\ndata: [DONE]\n".as_bytes();
        // Split right inside the 3-byte UTF-8 sequence for '你'
        let split_at = raw.iter().position(|&b| b >= 0x80).expect("utf8 byte") + 1;

        let part1 = decoder.push_bytes(&raw[..split_at]);
        assert_eq!(part1, Vec::<String>::new());

        let part2 = decoder.push_bytes(&raw[split_at..]);
        assert_eq!(
            part2,
            vec!["data: {\"hello\":\"你好Rust\"}", "data: [DONE]"]
        );
        assert!(decoder.flush_remaining().is_none());
    }

    #[test]
    fn parse_stream_handles_reasoning_text_tool_calls_and_cached_usage() {
        let sse_lines = [
            ": keep-alive",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think \"}}]}",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Checking file.\"}}]}",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Cargo.toml\\\"}\"}}]}}]}",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":50,\"prompt_tokens_details\":{\"cached_tokens\":850},\"completion_tokens_details\":{\"reasoning_tokens\":20}}}",
            "data: [DONE]",
        ];

        let mut events = Vec::new();
        let mut saw_done_sentinel = false;
        for line in sse_lines {
            if let Some(mut batch) = parse_sse_line(line).expect("parse sse") {
                events.append(&mut batch);
            } else {
                saw_done_sentinel = true;
                break;
            }
        }

        assert!(saw_done_sentinel);
        assert_eq!(
            events,
            vec![
                StreamEvent::ThinkingDelta {
                    text: "think ".into()
                },
                StreamEvent::TextDelta {
                    text: "Checking file.".into()
                },
                StreamEvent::ToolCallStart {
                    index: 0,
                    id: "call_1".into(),
                    name: "read".into(),
                },
                StreamEvent::ToolCallArgsDelta {
                    index: 0,
                    json_fragment: "{\"path\":".into(),
                },
                StreamEvent::ToolCallArgsDelta {
                    index: 0,
                    json_fragment: "\"Cargo.toml\"}".into(),
                },
                StreamEvent::Done {
                    reason: FinishReason::ToolCalls,
                },
                StreamEvent::Usage(TokenUsage {
                    input_tokens: 1000,
                    output_tokens: 50,
                    cache_read_tokens: 850,
                    cache_write_tokens: 0,
                    reasoning_tokens: 20,
                }),
            ]
        );
    }

    #[test]
    fn build_request_body_serializes_system_tools_and_tool_results() {
        let req = ProviderRequest {
            model: "deepseek-chat".into(),
            system: Some("You are OrangeAgent.".into()),
            messages: vec![
                Message::user("read Cargo.toml"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolCall {
                        id: "call_1".into(),
                        name: "read".into(),
                        arguments: json!({ "path": "Cargo.toml" }),
                    }],
                },
                Message::tool_result("call_1", "[workspace]", false),
            ],
            tools: vec![ToolDefinition {
                name: "read".into(),
                description: "Read a file".into(),
                parameters: json!({ "type": "object" }),
            }],
            max_tokens: Some(2048),
            temperature: Some(0.0),
        };

        let body = build_request_body(&req);
        assert_eq!(body["model"], "deepseek-chat");
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(
            body["messages"][2]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"Cargo.toml\"}"
        );
        assert_eq!(body["messages"][3]["role"], "tool");
        assert_eq!(body["messages"][3]["tool_call_id"], "call_1");
        assert_eq!(body["tools"][0]["function"]["name"], "read");
    }

    #[test]
    fn classify_http_error_detects_context_overflow_and_rate_limit() {
        let overflow =
            classify_http_error(400, r#"{"error":{"code":"context_length_exceeded"}}"#, None);
        assert_eq!(overflow.verdict().category, ErrorCategory::ContextOverflow);
        assert!(overflow.verdict().should_compress);

        let limited = classify_http_error(429, "rate limit reached", Some(2_000));
        assert_eq!(limited.verdict().category, ErrorCategory::RateLimit);
        assert_eq!(limited.verdict().retry_after_ms, Some(2_000));

        let auth = classify_http_error(401, "invalid api key", None);
        assert_eq!(auth.verdict().category, ErrorCategory::Auth);
        assert!(!auth.verdict().retryable);
    }
}
