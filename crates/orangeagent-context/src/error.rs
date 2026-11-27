use orangeagent_ai::AiError;
use thiserror::Error;

/// Structured errors returned by `orangeagent-context`.
#[derive(Debug, Error)]
pub enum ContextError {
    #[error("context I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("provider error during compaction: {0}")]
    Ai(#[from] AiError),

    #[error("tool_call closure invariant violated: {0}")]
    ClosureViolation(String),

    #[error("context operation cancelled")]
    Cancelled,
}
