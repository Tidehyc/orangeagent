use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use orangeagent_ai::ToolDefinition;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

/// Scheduling contract declared by each [`Tool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConcurrencyMode {
    /// Pure read-only tool (`read`, `grep`, `glob`) safe to execute concurrently via `join_all`.
    ReadOnlyParallel,
    /// State-mutating or interactive tool (`write`, `edit`, `bash`, `ask_user`) that must run
    /// strictly sequentially in the order emitted by the model.
    ExclusiveSequential,
}

/// Per-invocation execution context passed into [`Tool::execute`].
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub turn_id: String,
    pub call_id: String,
    pub cancel: CancellationToken,
}

/// Normalized result returned by a [`Tool`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
}

impl ToolOutput {
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// Executable tool contract consumed by the ReAct loop.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Unique tool name exposed to the LLM (e.g. `"read"`, `"bash"`).
    fn name(&self) -> &str;

    /// Concise description included in the static tool prefix.
    fn description(&self) -> &str;

    /// JSON Schema describing accepted arguments.
    fn parameters_schema(&self) -> Value;

    /// Concurrency classification. Defaults to [`ConcurrencyMode::ExclusiveSequential`]
    /// so newly added tools are write-safe unless explicitly opted into parallel execution.
    fn concurrency_mode(&self) -> ConcurrencyMode {
        ConcurrencyMode::ExclusiveSequential
    }

    /// Export this tool's LLM wire definition.
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters_schema(),
        }
    }

    /// Execute the tool with parsed JSON arguments.
    async fn execute(&self, args: Value, ctx: &ToolContext) -> ToolOutput;
}

/// Deterministic registry of available [`Tool`] implementations.
///
/// Uses [`BTreeMap`] rather than `HashMap` so [`ToolRegistry::definitions`] always
/// emits tools in identical sorted order across runs — preventing Prompt Cache invalidation.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or replace) a tool by its [`Tool::name`].
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).map(Arc::clone)
    }

    /// Return tool definitions in deterministic alphabetical order for Prompt Cache stability.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    struct DummyTool {
        name: &'static str,
        mode: ConcurrencyMode,
    }

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &str {
            self.name
        }

        fn description(&self) -> &str {
            "dummy tool"
        }

        fn parameters_schema(&self) -> Value {
            json!({ "type": "object" })
        }

        fn concurrency_mode(&self) -> ConcurrencyMode {
            self.mode
        }

        async fn execute(&self, _args: Value, _ctx: &ToolContext) -> ToolOutput {
            ToolOutput::ok(self.name)
        }
    }

    #[test]
    fn registry_emits_definitions_in_deterministic_sorted_order() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(DummyTool {
            name: "write",
            mode: ConcurrencyMode::ExclusiveSequential,
        }));
        reg.register(Arc::new(DummyTool {
            name: "bash",
            mode: ConcurrencyMode::ExclusiveSequential,
        }));
        reg.register(Arc::new(DummyTool {
            name: "read",
            mode: ConcurrencyMode::ReadOnlyParallel,
        }));

        let names: Vec<String> = reg.definitions().into_iter().map(|d| d.name).collect();
        assert_eq!(names, vec!["bash", "read", "write"]);
        assert_eq!(
            reg.get("read").expect("read tool").concurrency_mode(),
            ConcurrencyMode::ReadOnlyParallel
        );
    }
}
