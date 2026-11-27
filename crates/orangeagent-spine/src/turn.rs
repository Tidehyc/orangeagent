use std::path::PathBuf;

use async_trait::async_trait;
use orangeagent_core::{SteerQueue, TurnOutcome};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// Source of a scheduled turn, used by [`crate::OriginPools`] to isolate
/// foreground user interactivity from background automation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    User,
    Cron,
    Subagent,
}

impl Origin {
    /// Whether this turn originates from an interactive foreground user.
    pub fn is_user(self) -> bool {
        matches!(self, Self::User)
    }
}

/// How [`crate::SpineScheduler`] handles a turn submitted to a session that
/// already has an active turn running in its lane.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BusyPolicy {
    /// Queue behind the active turn (default).
    #[default]
    Append,
    /// Inject into the active turn's [`SteerQueue`] before the next LLM call;
    /// if the active turn finishes before draining it, smoothly degrade into an
    /// `Append` turn so the message is never lost.
    Inject,
    /// Cancel the active turn cooperatively and jump ahead of normal pending turns.
    Interrupt,
}

/// Incoming turn submission dispatched to [`crate::SpineScheduler`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnRequest {
    pub session_id: String,
    pub origin: Origin,
    pub text: String,
    #[serde(default)]
    pub busy_policy: BusyPolicy,
}

impl TurnRequest {
    pub fn user(session_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            origin: Origin::User,
            text: text.into(),
            busy_policy: BusyPolicy::Append,
        }
    }

    pub fn cron(session_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            origin: Origin::Cron,
            text: text.into(),
            busy_policy: BusyPolicy::Append,
        }
    }

    pub fn subagent(session_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            origin: Origin::Subagent,
            text: text.into(),
            busy_policy: BusyPolicy::Append,
        }
    }

    #[must_use]
    pub fn with_busy_policy(mut self, policy: BusyPolicy) -> Self {
        self.busy_policy = policy;
        self
    }
}

/// Terminal receipt returned to the caller awaiting a [`TurnHandle`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TurnReceipt {
    /// The request ran (either directly or after degrading from `Inject`) as a turn.
    Executed {
        turn_id: String,
        outcome: TurnOutcome,
    },
    /// The request was consumed mid-turn by the active turn's [`SteerQueue`].
    Injected { target_turn_id: String },
}

/// Execution context handed to a [`TurnExecutor`] for one scheduled turn.
#[derive(Debug, Clone)]
pub struct TurnContext {
    pub turn_id: String,
    pub session_id: String,
    pub origin: Origin,
    pub text: String,
    pub steer: SteerQueue,
    pub cancel: CancellationToken,
}

/// Abstraction decoupling [`crate::SpineScheduler`] from how a turn actually runs
/// (production `AgentLoop`, VCR replay, or deterministic concurrency mocks).
#[async_trait]
pub trait TurnExecutor: Send + Sync {
    async fn execute(&self, ctx: TurnContext) -> TurnOutcome;
}

/// Handle returned immediately upon submitting a [`TurnRequest`].
#[derive(Debug)]
pub struct TurnHandle {
    session_id: String,
    rx: oneshot::Receiver<Result<TurnReceipt, SpineError>>,
}

impl TurnHandle {
    pub(crate) fn new(
        session_id: String,
        rx: oneshot::Receiver<Result<TurnReceipt, SpineError>>,
    ) -> Self {
        Self { session_id, rx }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Await the scheduling outcome of this request.
    pub async fn wait(self) -> Result<TurnReceipt, SpineError> {
        self.rx.await.map_err(|_| SpineError::LaneClosed)?
    }
}

/// Structured errors produced by the scheduling and cross-process locking layer.
#[derive(Debug, Error)]
pub enum SpineError {
    #[error("scheduler is sealed for shutdown and no longer accepts new turns")]
    Sealed,

    #[error("turn cancelled before completion")]
    Cancelled,

    #[error("session lane worker exited before delivering a receipt")]
    LaneClosed,

    #[error("file lock is already held by another process: {path}")]
    LockContended { path: PathBuf },

    #[error("file lock I/O failure at {path}: {source}")]
    LockIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}
