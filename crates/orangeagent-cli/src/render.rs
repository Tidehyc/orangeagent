use std::io::Write;

use orangeagent_core::{AgentEvent, TurnOutcome};
use tokio::sync::mpsc::UnboundedReceiver;

/// Output formatting mode for `orangeagent run`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Human-readable streaming text on stdout + tool/status metadata on stderr.
    Human,
    /// Machine-readable newline-delimited JSON (`AgentEvent` JSONL) on stdout.
    JsonLines,
}

/// Drain `rx` until closed, rendering every [`AgentEvent`] to `stdout` / `stderr`.
pub async fn render_event_stream<WOut, WErr>(
    mut rx: UnboundedReceiver<AgentEvent>,
    mode: OutputMode,
    mut stdout: WOut,
    mut stderr: WErr,
) -> std::io::Result<()>
where
    WOut: Write + Send,
    WErr: Write + Send,
{
    let mut text_started = false;
    let mut thinking_open = false;

    while let Some(event) = rx.recv().await {
        match mode {
            OutputMode::JsonLines => {
                let line = serde_json::to_string(&event)
                    .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
                writeln!(stdout, "{line}")?;
                stdout.flush()?;
            }
            OutputMode::Human => {
                render_human_event(
                    &event,
                    &mut stdout,
                    &mut stderr,
                    &mut text_started,
                    &mut thinking_open,
                )?;
            }
        }
    }

    if mode == OutputMode::Human && text_started {
        writeln!(stdout)?;
        stdout.flush()?;
    }

    Ok(())
}

fn render_human_event<WOut: Write, WErr: Write>(
    event: &AgentEvent,
    stdout: &mut WOut,
    stderr: &mut WErr,
    text_started: &mut bool,
    thinking_open: &mut bool,
) -> std::io::Result<()> {
    match event {
        AgentEvent::ThinkingDelta { text } => {
            if !*thinking_open {
                write!(stderr, "[thinking] ")?;
                *thinking_open = true;
            }
            write!(stderr, "{text}")?;
            stderr.flush()?;
        }
        AgentEvent::TextDelta { text } => {
            if *thinking_open {
                writeln!(stderr)?;
                *thinking_open = false;
            }
            write!(stdout, "{text}")?;
            stdout.flush()?;
            *text_started = true;
        }
        AgentEvent::ToolCallStarted {
            name, arguments, ..
        } => {
            close_open_streams(stdout, stderr, text_started, thinking_open)?;
            writeln!(stderr, "[tool] {name} {arguments}")?;
            stderr.flush()?;
        }
        AgentEvent::ToolCallFinished {
            name,
            is_error,
            output_preview,
            ..
        } => {
            let status = if *is_error { "ERROR" } else { "ok" };
            let first_line = output_preview.lines().next().unwrap_or("");
            writeln!(stderr, "[tool] {name} -> {status}: {first_line}")?;
            stderr.flush()?;
        }
        AgentEvent::RecoveryTriggered { attempt, reason } => {
            close_open_streams(stdout, stderr, text_started, thinking_open)?;
            writeln!(stderr, "[recovery #{attempt}] {reason}")?;
            stderr.flush()?;
        }
        AgentEvent::TurnFinished { outcome } => {
            close_open_streams(stdout, stderr, text_started, thinking_open)?;
            render_outcome_footer(outcome, stderr)?;
        }
        AgentEvent::TurnStarted { .. }
        | AgentEvent::IterationStarted { .. }
        | AgentEvent::UsageUpdated { .. } => {}
    }
    Ok(())
}

fn close_open_streams<WOut: Write, WErr: Write>(
    stdout: &mut WOut,
    stderr: &mut WErr,
    text_started: &mut bool,
    thinking_open: &mut bool,
) -> std::io::Result<()> {
    if *thinking_open {
        writeln!(stderr)?;
        *thinking_open = false;
    }
    if *text_started {
        writeln!(stdout)?;
        stdout.flush()?;
        *text_started = false;
    }
    Ok(())
}

fn render_outcome_footer<WErr: Write>(
    outcome: &TurnOutcome,
    stderr: &mut WErr,
) -> std::io::Result<()> {
    match outcome {
        TurnOutcome::Completed {
            iterations,
            total_usage,
            ..
        } => {
            let cache_pct = total_usage.cache_hit_ratio() * 100.0;
            writeln!(
                stderr,
                "[done] iterations={iterations} tokens(in={}, cached={} [{cache_pct:.1}%], out={})",
                total_usage.input_tokens, total_usage.cache_read_tokens, total_usage.output_tokens
            )?;
        }
        TurnOutcome::Interrupted {
            reason,
            iterations,
            total_usage,
        } => {
            writeln!(
                stderr,
                "[interrupted] reason={reason} iterations={iterations} tokens(in={}, out={})",
                total_usage.input_tokens, total_usage.output_tokens
            )?;
        }
        TurnOutcome::Failed {
            error,
            iterations,
            total_usage,
        } => {
            writeln!(
                stderr,
                "[failed] error={error} iterations={iterations} tokens(in={}, out={})",
                total_usage.input_tokens, total_usage.output_tokens
            )?;
        }
    }
    stderr.flush()
}
