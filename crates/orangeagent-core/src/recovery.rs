use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Bounded self-healing limits for transient LLM output glitches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryConfig {
    /// Maximum consecutive empty assistant responses before failing the turn.
    pub max_empty_retries: usize,
    /// Maximum consecutive malformed tool-argument JSON attempts before failing the turn.
    pub max_malformed_json_retries: usize,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            max_empty_retries: 2,
            max_malformed_json_retries: 2,
        }
    }
}

/// Tracks consecutive recovery attempts within a single turn.
///
/// Counters reset as soon as the model produces valid progress so long multi-step
/// turns are never penalized for an earlier transient glitch.
#[derive(Debug, Clone)]
pub struct RecoveryState {
    config: RecoveryConfig,
    empty_attempts: usize,
    malformed_json_attempts: usize,
}

impl RecoveryState {
    pub fn new(config: RecoveryConfig) -> Self {
        Self {
            config,
            empty_attempts: 0,
            malformed_json_attempts: 0,
        }
    }

    /// Clear consecutive failure counters after valid model progress.
    pub fn reset(&mut self) {
        self.empty_attempts = 0;
        self.malformed_json_attempts = 0;
    }

    /// Record an empty assistant response (no text and no tool calls).
    ///
    /// Returns `Some((attempt, steer_prompt))` if a retry is permitted, or `None`
    /// once `max_empty_retries` is exhausted.
    pub fn on_empty_response(&mut self) -> Option<(usize, String)> {
        if self.empty_attempts >= self.config.max_empty_retries {
            return None;
        }
        self.empty_attempts += 1;
        let prompt = "Your previous response was empty (no text and no tool calls). \
                      Please either invoke a tool or provide a direct answer."
            .to_string();
        Some((self.empty_attempts, prompt))
    }

    /// Record a malformed JSON tool-call argument payload.
    ///
    /// Returns `Some((attempt, error_feedback))` to feed back as a `ToolOutput::error`
    /// so the model can self-correct, or `None` once the retry budget is exhausted.
    pub fn on_malformed_tool_args(
        &mut self,
        tool_name: &str,
        parse_err: &str,
    ) -> Option<(usize, String)> {
        if self.malformed_json_attempts >= self.config.max_malformed_json_retries {
            return None;
        }
        self.malformed_json_attempts += 1;
        let feedback = format!(
            "Invalid JSON arguments for tool `{tool_name}`: {parse_err}. \
             Please retry with a valid JSON object matching the tool schema."
        );
        Some((self.malformed_json_attempts, feedback))
    }
}

/// Parse raw tool-call argument text into a JSON object [`Value`].
///
/// Empty/whitespace-only strings are normalized to `{}` (some OpenAI-compatible
/// endpoints omit `{}` for zero-argument tools). Non-object JSON values such as
/// bare strings or arrays are rejected because tool schemas always expect an object.
pub fn parse_tool_arguments(raw: &str) -> Result<Value, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Value::Object(Map::new()));
    }

    let parsed: Value = serde_json::from_str(trimmed).map_err(|err| err.to_string())?;
    if !parsed.is_object() {
        return Err(format!("expected JSON object, got {parsed}"));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parse_tool_arguments_handles_empty_valid_and_invalid_inputs() {
        assert_eq!(parse_tool_arguments("").expect("empty"), json!({}));
        assert_eq!(parse_tool_arguments("   ").expect("whitespace"), json!({}));
        assert_eq!(
            parse_tool_arguments(r#"{"path":"src/lib.rs"}"#).expect("valid object"),
            json!({ "path": "src/lib.rs" })
        );
        assert!(parse_tool_arguments(r#"{"path": "#).is_err());
        assert!(parse_tool_arguments("[1, 2, 3]").is_err());
    }

    #[test]
    fn recovery_state_enforces_budget_and_resets_on_progress() {
        let mut state = RecoveryState::new(RecoveryConfig {
            max_empty_retries: 2,
            max_malformed_json_retries: 1,
        });

        assert_eq!(state.on_empty_response().map(|(n, _)| n), Some(1));
        assert_eq!(state.on_empty_response().map(|(n, _)| n), Some(2));
        assert!(state.on_empty_response().is_none());

        state.reset();
        assert_eq!(state.on_empty_response().map(|(n, _)| n), Some(1));

        assert_eq!(
            state.on_malformed_tool_args("read", "EOF").map(|(n, _)| n),
            Some(1)
        );
        assert!(state.on_malformed_tool_args("read", "EOF").is_none());
    }
}
