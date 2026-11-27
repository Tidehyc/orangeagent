use orangeagent_ai::{ContentBlock, Message, ToolDefinition};

const MESSAGE_FRAMING_TOKENS: u64 = 4;

/// Deterministic CJK-safe token count heuristic for pre-flight budgeting
/// and protected-tail sizing.
pub fn estimate_text_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    let byte_est = u64::try_from(text.len().div_ceil(4)).unwrap_or(u64::MAX);
    let char_est = u64::try_from(text.chars().count().div_ceil(3)).unwrap_or(u64::MAX);
    byte_est.max(char_est).max(1)
}

/// Estimate tokens consumed by a single [`ContentBlock`].
pub fn estimate_block_tokens(block: &ContentBlock) -> u64 {
    match block {
        ContentBlock::Text { text } | ContentBlock::Thinking { text, .. } => {
            estimate_text_tokens(text)
        }
        ContentBlock::ToolCall {
            name, arguments, ..
        } => {
            let args_str = serde_json::to_string(arguments).unwrap_or_default();
            estimate_text_tokens(name) + estimate_text_tokens(&args_str) + 8
        }
        ContentBlock::ToolResult { content, .. } => estimate_text_tokens(content) + 6,
    }
}

/// Estimate tokens consumed by a single [`Message`], including role/framing overhead.
pub fn estimate_message_tokens(msg: &Message) -> u64 {
    let body: u64 = msg.content.iter().map(estimate_block_tokens).sum();
    MESSAGE_FRAMING_TOKENS + body
}

/// Estimate total tokens across a slice of [`Message`]s.
pub fn estimate_messages_tokens(messages: &[Message]) -> u64 {
    messages.iter().map(estimate_message_tokens).sum()
}

/// Estimate tokens consumed by the static [`ToolDefinition`] schema list.
pub fn estimate_tools_tokens(tools: &[ToolDefinition]) -> u64 {
    tools
        .iter()
        .map(|tool| {
            let schema_str = serde_json::to_string(&tool.parameters).unwrap_or_default();
            estimate_text_tokens(&tool.name)
                + estimate_text_tokens(&tool.description)
                + estimate_text_tokens(&schema_str)
                + 12
        })
        .sum()
}

/// Estimate total prompt tokens for `(system, tools, messages)`.
pub fn estimate_request_tokens(
    system: Option<&str>,
    tools: &[ToolDefinition],
    messages: &[Message],
) -> u64 {
    let sys_tokens = system.map_or(0, |s| estimate_text_tokens(s) + MESSAGE_FRAMING_TOKENS);
    sys_tokens + estimate_tools_tokens(tools) + estimate_messages_tokens(messages)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn token_estimator_handles_empty_ascii_cjk_and_tool_requests() {
        assert_eq!(estimate_text_tokens(""), 0);
        assert!(estimate_text_tokens("hi") >= 1);
        assert!(estimate_text_tokens("你好世界测试文本") >= 3);

        let tools = vec![ToolDefinition {
            name: "read".into(),
            description: "Read a UTF-8 file".into(),
            parameters: json!({ "type": "object" }),
        }];
        let msgs = vec![Message::user("inspect Cargo.toml")];
        let total = estimate_request_tokens(Some("You are OrangeAgent."), &tools, &msgs);
        assert!(total > estimate_messages_tokens(&msgs));
    }
}
