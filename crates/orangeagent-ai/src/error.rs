use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Coarse-grained error category used by upper layers (`orangeagent-core` / `orangeagent-context`)
/// to decide recovery strategies without parsing provider-specific error strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    RateLimit,
    ContextOverflow,
    Auth,
    InvalidRequest,
    TransientNetwork,
    ServerError,
    Cancelled,
    Protocol,
}

/// Actionable decision derived from an [`AiError`].
///
/// Instead of scattering `if status == 429` across the Agent Loop, every provider error
/// maps into one `ErrorVerdict` that tells the loop whether to retry, trigger context
/// compaction, or abort immediately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorVerdict {
    pub category: ErrorCategory,
    pub retryable: bool,
    pub should_compress: bool,
    pub retry_after_ms: Option<u64>,
}

/// Unified error type for all LLM provider and VCR operations.
#[derive(Debug, Error)]
pub enum AiError {
    #[error("rate limited (retry_after_ms={retry_after_ms:?}): {message}")]
    RateLimited {
        retry_after_ms: Option<u64>,
        message: String,
    },

    #[error("context window overflow: {message}")]
    ContextOverflow { message: String },

    #[error("authentication failed: {0}")]
    Authentication(String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("http error {status}: {message}")]
    Http {
        status: u16,
        message: String,
        retry_after_ms: Option<u64>,
    },

    #[error("network transport error: {0}")]
    Network(String),

    #[error("protocol / stream decode error: {0}")]
    Protocol(String),

    #[error("request cancelled")]
    Cancelled,

    #[error("vcr playback/record error: {0}")]
    Vcr(String),
}

impl AiError {
    /// Translate this error into an actionable [`ErrorVerdict`] for retry/compaction loops.
    pub fn verdict(&self) -> ErrorVerdict {
        match self {
            Self::RateLimited { retry_after_ms, .. } => ErrorVerdict {
                category: ErrorCategory::RateLimit,
                retryable: true,
                should_compress: false,
                retry_after_ms: *retry_after_ms,
            },
            Self::ContextOverflow { .. } => ErrorVerdict {
                category: ErrorCategory::ContextOverflow,
                retryable: true,
                should_compress: true,
                retry_after_ms: None,
            },
            Self::Authentication(_) => ErrorVerdict {
                category: ErrorCategory::Auth,
                retryable: false,
                should_compress: false,
                retry_after_ms: None,
            },
            Self::InvalidRequest(_) => ErrorVerdict {
                category: ErrorCategory::InvalidRequest,
                retryable: false,
                should_compress: false,
                retry_after_ms: None,
            },
            Self::Http {
                status,
                retry_after_ms,
                ..
            } => match *status {
                401 | 403 => ErrorVerdict {
                    category: ErrorCategory::Auth,
                    retryable: false,
                    should_compress: false,
                    retry_after_ms: None,
                },
                429 => ErrorVerdict {
                    category: ErrorCategory::RateLimit,
                    retryable: true,
                    should_compress: false,
                    retry_after_ms: *retry_after_ms,
                },
                500..=599 => ErrorVerdict {
                    category: ErrorCategory::ServerError,
                    retryable: true,
                    should_compress: false,
                    retry_after_ms: *retry_after_ms,
                },
                _ => ErrorVerdict {
                    category: ErrorCategory::InvalidRequest,
                    retryable: false,
                    should_compress: false,
                    retry_after_ms: None,
                },
            },
            Self::Network(_) => ErrorVerdict {
                category: ErrorCategory::TransientNetwork,
                retryable: true,
                should_compress: false,
                retry_after_ms: None,
            },
            Self::Protocol(_) | Self::Vcr(_) => ErrorVerdict {
                category: ErrorCategory::Protocol,
                retryable: false,
                should_compress: false,
                retry_after_ms: None,
            },
            Self::Cancelled => ErrorVerdict {
                category: ErrorCategory::Cancelled,
                retryable: false,
                should_compress: false,
                retry_after_ms: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_overflow_requests_compaction() {
        let err = AiError::ContextOverflow {
            message: "prompt exceeds 200k tokens".into(),
        };
        let verdict = err.verdict();
        assert_eq!(verdict.category, ErrorCategory::ContextOverflow);
        assert!(verdict.retryable);
        assert!(verdict.should_compress);
    }

    #[test]
    fn http_status_maps_to_expected_verdicts() {
        let rate_limit = AiError::Http {
            status: 429,
            message: "slow down".into(),
            retry_after_ms: Some(1_500),
        };
        assert_eq!(
            rate_limit.verdict(),
            ErrorVerdict {
                category: ErrorCategory::RateLimit,
                retryable: true,
                should_compress: false,
                retry_after_ms: Some(1_500),
            }
        );

        let auth = AiError::Http {
            status: 401,
            message: "bad key".into(),
            retry_after_ms: None,
        };
        assert!(!auth.verdict().retryable);
        assert_eq!(auth.verdict().category, ErrorCategory::Auth);

        let server = AiError::Http {
            status: 503,
            message: "overloaded".into(),
            retry_after_ms: None,
        };
        assert!(server.verdict().retryable);
        assert_eq!(server.verdict().category, ErrorCategory::ServerError);
    }
}
