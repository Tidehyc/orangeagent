use orangeagent_ai::TokenUsage;
use serde::{Deserialize, Serialize};

/// Terminal result of executing a single agent turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TurnOutcome {
    /// The model finished normally without pending tool calls.
    Completed {
        final_text: String,
        iterations: usize,
        total_usage: TokenUsage,
    },
    /// The turn was cancelled cooperatively (e.g., user interrupt or hook short-circuit).
    Interrupted {
        reason: String,
        iterations: usize,
        total_usage: TokenUsage,
    },
    /// The turn aborted due to an unrecoverable error or iteration budget exhaustion.
    Failed {
        error: String,
        iterations: usize,
        total_usage: TokenUsage,
    },
}
