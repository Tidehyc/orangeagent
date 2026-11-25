use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Deterministic lifecycle checkpoints exposed to internal and external hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookPhase {
    BeforeTurn,
    BeforeModelCall,
    AfterModelCall,
    BeforeToolUse,
    AfterToolUse,
    AfterTurn,
}

/// Input context delivered to each [`AgentHook`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HookEvent {
    pub phase: HookPhase,
    pub turn_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    pub payload: Value,
}

/// Control decision returned by an [`AgentHook`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum HookDecision {
    /// Allow execution to continue unmodified.
    PassThrough,
    /// Replace the current stage payload (e.g., rewrite tool arguments or append LSP diagnostics).
    Modify(Value),
    /// Reject the action and feed `reason` back to the model as a tool error.
    Deny(String),
    /// Immediately terminate the turn with `final_text`.
    ShortCircuit(String),
}

/// Extensible lifecycle interceptor (implemented by approval guards, LSP feedback, shell hooks, etc.).
#[async_trait]
pub trait AgentHook: Send + Sync {
    fn name(&self) -> &str;

    async fn on_event(&self, event: &HookEvent) -> HookDecision;
}

/// Ordered pipeline of [`AgentHook`]s.
///
/// Chains `Modify` payloads sequentially across hooks and short-circuits immediately
/// on the first `Deny` or `ShortCircuit` decision.
#[derive(Default, Clone)]
pub struct CompositeHook {
    hooks: Vec<Arc<dyn AgentHook>>,
}

impl CompositeHook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, hook: Arc<dyn AgentHook>) {
        self.hooks.push(hook);
    }

    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// Run all registered hooks in order for `event`.
    pub async fn dispatch(&self, event: &HookEvent) -> HookDecision {
        let mut current = event.clone();
        let mut modified = false;

        for hook in &self.hooks {
            match hook.on_event(&current).await {
                HookDecision::PassThrough => {}
                HookDecision::Modify(next_payload) => {
                    current.payload = next_payload;
                    modified = true;
                }
                terminal @ (HookDecision::Deny(_) | HookDecision::ShortCircuit(_)) => {
                    return terminal;
                }
            }
        }

        if modified {
            HookDecision::Modify(current.payload)
        } else {
            HookDecision::PassThrough
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    struct AppendTagHook(&'static str);

    #[async_trait]
    impl AgentHook for AppendTagHook {
        fn name(&self) -> &str {
            self.0
        }

        async fn on_event(&self, event: &HookEvent) -> HookDecision {
            let mut next = event.payload.clone();
            if let Some(arr) = next.get_mut("tags").and_then(Value::as_array_mut) {
                arr.push(json!(self.0));
            }
            HookDecision::Modify(next)
        }
    }

    struct DenyBashHook;

    #[async_trait]
    impl AgentHook for DenyBashHook {
        fn name(&self) -> &str {
            "deny-bash"
        }

        async fn on_event(&self, event: &HookEvent) -> HookDecision {
            if event.tool_name.as_deref() == Some("bash") {
                HookDecision::Deny("bash blocked by policy".into())
            } else {
                HookDecision::PassThrough
            }
        }
    }

    #[tokio::test]
    async fn composite_hook_chains_modifications_and_stops_on_deny() {
        let mut chain = CompositeHook::new();
        chain.push(Arc::new(AppendTagHook("first")));
        chain.push(Arc::new(DenyBashHook));
        chain.push(Arc::new(AppendTagHook("second")));

        let read_event = HookEvent {
            phase: HookPhase::BeforeToolUse,
            turn_id: "turn_1".into(),
            tool_name: Some("read".into()),
            payload: json!({ "tags": [] }),
        };
        assert_eq!(
            chain.dispatch(&read_event).await,
            HookDecision::Modify(json!({ "tags": ["first", "second"] }))
        );

        let bash_event = HookEvent {
            phase: HookPhase::BeforeToolUse,
            turn_id: "turn_1".into(),
            tool_name: Some("bash".into()),
            payload: json!({ "tags": [] }),
        };
        assert_eq!(
            chain.dispatch(&bash_event).await,
            HookDecision::Deny("bash blocked by policy".into())
        );
    }
}
