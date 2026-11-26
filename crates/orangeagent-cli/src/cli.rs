use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

/// Pure-Rust Coding Agent & Harness CLI.
#[derive(Debug, Parser)]
#[command(
    name = "orangeagent",
    version,
    about = "Pure-Rust Coding Agent & Harness"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Execute a single non-interactive coding turn with streaming or `--json` output.
    Run(RunArgs),
}

/// Arguments for `orangeagent run`.
#[derive(Debug, Clone, Args)]
pub struct RunArgs {
    /// User instruction to execute.
    pub prompt: String,

    /// Workspace root directory exposed to coding tools.
    #[arg(short, long, default_value = ".")]
    pub workspace: PathBuf,

    /// Model name passed to the provider.
    #[arg(short, long, env = "ORANGEAGENT_MODEL", default_value = "gpt-4o-mini")]
    pub model: String,

    /// OpenAI Chat Completions compatible base URL.
    #[arg(
        long,
        env = "ORANGEAGENT_BASE_URL",
        default_value = "https://api.openai.com/v1"
    )]
    pub base_url: String,

    /// Bearer API key (falls back to `OPENAI_API_KEY` if omitted).
    #[arg(long, env = "ORANGEAGENT_API_KEY")]
    pub api_key: Option<String>,

    /// Optional custom system prompt (overrides the static default prompt).
    #[arg(long)]
    pub system: Option<String>,

    /// Maximum ReAct iterations before aborting.
    #[arg(long, default_value_t = 25)]
    pub max_iterations: usize,

    /// Optional JSONL session file path for loading prior history and persisting this turn.
    #[arg(short, long)]
    pub session: Option<PathBuf>,

    /// Emit newline-delimited JSON (`AgentEvent` JSONL) to stdout for headless evaluation.
    #[arg(long)]
    pub json: bool,

    /// Replay a recorded VCR JSONL cassette offline instead of making network requests.
    #[arg(long, conflicts_with = "vcr_record")]
    pub vcr_replay: Option<PathBuf>,

    /// Record live provider interactions into a VCR JSONL cassette file.
    #[arg(long, conflicts_with = "vcr_replay")]
    pub vcr_record: Option<PathBuf>,

    /// Enforce strict request-fingerprint matching when replaying a VCR cassette.
    #[arg(long, default_value_t = false)]
    pub vcr_strict_hash: bool,
}
