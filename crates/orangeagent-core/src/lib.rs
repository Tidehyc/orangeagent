//! `orangeagent-core` — Minimal event-driven ReAct loop, tool execution contracts,
//! lifecycle hooks, and structured turn outcomes.
//!
//! Strictly depends only on `orangeagent-ai` (never on filesystem, git, or UI).

pub mod events;
pub mod hook;
pub mod outcome;
pub mod tool;

pub use events::{AgentEvent, EventSink};
pub use hook::{AgentHook, CompositeHook, HookDecision, HookEvent, HookPhase};
pub use outcome::TurnOutcome;
pub use tool::{ConcurrencyMode, Tool, ToolContext, ToolOutput, ToolRegistry};
