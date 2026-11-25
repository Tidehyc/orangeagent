use orangeagent_ai::TokenUsage;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

use crate::TurnOutcome;

/// Strongly-typed runtime event stream consumed by TUI, CLI `--json`, logs, or IPC clients.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    TurnStarted {
        turn_id: String,
    },
    IterationStarted {
        iteration: usize,
    },
    ThinkingDelta {
        text: String,
    },
    TextDelta {
        text: String,
    },
    ToolCallStarted {
        call_id: String,
        name: String,
        arguments: Value,
    },
    ToolCallFinished {
        call_id: String,
        name: String,
        is_error: bool,
        output_preview: String,
    },
    UsageUpdated {
        usage: TokenUsage,
    },
    RecoveryTriggered {
        attempt: usize,
        reason: String,
    },
    TurnFinished {
        outcome: TurnOutcome,
    },
}

/// Non-blocking channel handle for emitting [`AgentEvent`]s from the ReAct loop.
pub type EventSink = UnboundedSender<AgentEvent>;
