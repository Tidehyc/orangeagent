//! `orangeagent-harness` — Concrete coding tools (`read`, `grep`, `glob`, `write`, `edit`, `bash`,
//! and later `subagent`, `ask_user`) bound to a sandboxed workspace root.

pub mod tools;
pub mod workspace;

use std::sync::Arc;

use orangeagent_core::ToolRegistry;

pub use tools::bash::BashTool;
pub use tools::edit::EditTool;
pub use tools::glob::GlobTool;
pub use tools::grep::GrepTool;
pub use tools::read::ReadTool;
pub use tools::write::WriteTool;
pub use workspace::{ReadFileTracker, WorkspaceContext};

/// Register the read-only inspection and search tools (`read`, `grep`, `glob`) into `registry`.
pub fn register_readonly_tools(registry: &mut ToolRegistry, workspace: WorkspaceContext) {
    registry.register(Arc::new(ReadTool::new(workspace.clone())));
    registry.register(Arc::new(GrepTool::new(workspace.clone())));
    registry.register(Arc::new(GlobTool::new(workspace)));
}

/// Register all 6 core coding tools (`read`, `grep`, `glob`, `write`, `edit`, `bash`) into `registry`.
pub fn register_core_tools(registry: &mut ToolRegistry, workspace: WorkspaceContext) {
    register_readonly_tools(registry, workspace.clone());
    registry.register(Arc::new(WriteTool::new(workspace.clone())));
    registry.register(Arc::new(EditTool::new(workspace.clone())));
    registry.register(Arc::new(BashTool::new(workspace)));
}
