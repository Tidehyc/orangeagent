use std::process::ExitCode;

use clap::Parser as _;
use orangeagent_cli::{Cli, Commands, execute_run, exit_code_for_outcome};
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();

    let cancel = CancellationToken::new();
    let cancel_for_signal = cancel.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            cancel_for_signal.cancel();
        }
    });

    match cli.command {
        Commands::Run(args) => {
            match execute_run(&args, std::io::stdout(), std::io::stderr(), cancel).await {
                Ok(report) => exit_code_for_outcome(&report.outcome),
                Err(err) => {
                    eprintln!("orangeagent error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}
