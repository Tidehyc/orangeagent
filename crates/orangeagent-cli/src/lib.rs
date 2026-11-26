//! `orangeagent-cli` — Command-line frontend (`orangeagent run` and later TUI / Daemon / ACP).

pub mod cli;
pub mod render;
pub mod runner;

pub use cli::{Cli, Commands, RunArgs};
pub use render::OutputMode;
pub use runner::{DEFAULT_SYSTEM_PROMPT, RunReport, execute_run, exit_code_for_outcome};
