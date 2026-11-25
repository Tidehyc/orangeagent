//! `orangeagent-harness` — Concrete coding tools (`read`, `grep`, `glob`, and later
//! `write`, `edit`, `bash`, `subagent`) bound to a sandboxed workspace root.

pub mod tools;
pub mod workspace;

use std::sync::Arc;

use orangeagent_core::ToolRegistry;

pub use tools::glob::GlobTool;
pub use tools::grep::GrepTool;
pub use tools::read::ReadTool;
pub use workspace::{ReadFileTracker, WorkspaceContext};

/// Register the read-only inspection and search tools (`read`, `grep`, `glob`) into `registry`.
pub fn register_readonly_tools(registry: &mut ToolRegistry, workspace: WorkspaceContext) {
    registry.register(Arc::new(ReadTool::new(workspace.clone())));
    registry.register(Arc::new(GrepTool::new(workspace.clone())));
    registry.register(Arc::new(GlobTool::new(workspace)));
}
