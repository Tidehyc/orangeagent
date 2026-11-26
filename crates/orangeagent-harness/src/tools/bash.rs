use std::collections::VecDeque;
use std::fmt::Write as _;
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use orangeagent_core::{ConcurrencyMode, Tool, ToolContext, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt as _};
use tokio::process::{Child, Command};

use crate::WorkspaceContext;

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_TIMEOUT_SECS: u64 = 600;
const DEFAULT_MAX_OUTPUT_BYTES: usize = 30 * 1024;
const KILL_GRACE_MS: u64 = 150;
const DRAIN_AFTER_KILL_MS: u64 = 250;

/// Shell execution tool (`ConcurrencyMode::ExclusiveSequential`).
///
/// Runs commands inside the workspace with configurable timeout, dedicated Unix process-group
/// (`PGID`) isolation so timeout/cancellation terminates all descendant processes without
/// leaving orphans, and bounded head+tail output truncation.
pub struct BashTool {
    workspace: WorkspaceContext,
    default_timeout: Duration,
    max_output_bytes: usize,
}

impl BashTool {
    pub fn new(workspace: WorkspaceContext) -> Self {
        Self {
            workspace,
            default_timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
        }
    }

    #[must_use]
    pub fn with_default_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    #[must_use]
    pub fn with_max_output_bytes(mut self, max_output_bytes: usize) -> Self {
        self.max_output_bytes = max_output_bytes.max(128);
        self
    }
}

#[derive(Debug, Deserialize)]
struct BashArgs {
    command: String,
    #[serde(default)]
    workdir: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

struct CapturedStream {
    text: String,
    total_bytes: usize,
    truncated: bool,
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command inside the workspace with timeout, process-group cleanup, and bounded output."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command executed via `bash -c`."
                },
                "workdir": {
                    "type": "string",
                    "description": "Optional working directory inside the workspace (default: workspace root)."
                },
                "timeout_secs": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_TIMEOUT_SECS,
                    "description": "Execution timeout in seconds (default: 120, max: 600)."
                }
            },
            "required": ["command"]
        })
    }

    fn concurrency_mode(&self) -> ConcurrencyMode {
        ConcurrencyMode::ExclusiveSequential
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> ToolOutput {
        if ctx.cancel.is_cancelled() {
            return ToolOutput::error("bash cancelled");
        }

        let parsed: BashArgs = match serde_json::from_value(args) {
            Ok(v) => v,
            Err(err) => return ToolOutput::error(format!("invalid arguments for `bash`: {err}")),
        };

        let workdir = match &parsed.workdir {
            Some(raw) => match self.workspace.resolve_existing_path(raw) {
                Ok(path) if path.is_dir() => path,
                Ok(_) => return ToolOutput::error(format!("workdir `{raw}` is not a directory")),
                Err(err) => return ToolOutput::error(err),
            },
            None => self.workspace.root().to_path_buf(),
        };

        let timeout = parsed.timeout_secs.map_or(self.default_timeout, |secs| {
            Duration::from_secs(secs.clamp(1, MAX_TIMEOUT_SECS))
        });

        self.run_command(&parsed.command, &workdir, timeout, ctx)
            .await
    }
}

impl BashTool {
    async fn run_command(
        &self,
        command: &str,
        workdir: &std::path::Path,
        timeout: Duration,
        ctx: &ToolContext,
    ) -> ToolOutput {
        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg(command)
            .current_dir(workdir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        #[cfg(unix)]
        cmd.process_group(0);

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(err) => return ToolOutput::error(format!("failed to spawn bash: {err}")),
        };

        let child_pid = child.id();
        let per_stream_cap = (self.max_output_bytes / 2).max(64);
        let stdout_task = child
            .stdout
            .take()
            .map(|r| tokio::spawn(drain_capped_stream(r, per_stream_cap)));
        let stderr_task = child
            .stderr
            .take()
            .map(|r| tokio::spawn(drain_capped_stream(r, per_stream_cap)));

        let wait_outcome = tokio::select! {
            res = child.wait() => WaitOutcome::Exited(res),
            () = tokio::time::sleep(timeout) => WaitOutcome::TimedOut,
            () = ctx.cancel.cancelled() => WaitOutcome::Cancelled,
        };

        match wait_outcome {
            WaitOutcome::Exited(Ok(status)) => {
                let stdout = collect_stream(stdout_task, None).await;
                let stderr = collect_stream(stderr_task, None).await;
                let code = status.code().unwrap_or(-1);
                let body = format_output(code, None, &stdout, &stderr);
                if status.success() {
                    ToolOutput::ok(body)
                } else {
                    ToolOutput::error(body)
                }
            }
            WaitOutcome::Exited(Err(err)) => {
                kill_process_group(child_pid, &mut child).await;
                ToolOutput::error(format!("failed waiting for bash process: {err}"))
            }
            WaitOutcome::TimedOut => {
                kill_process_group(child_pid, &mut child).await;
                let grace = Some(Duration::from_millis(DRAIN_AFTER_KILL_MS));
                let stdout = collect_stream(stdout_task, grace).await;
                let stderr = collect_stream(stderr_task, grace).await;
                let reason = format!(
                    "command timed out after {}s (process group terminated)",
                    timeout.as_secs()
                );
                ToolOutput::error(format_output(-1, Some(&reason), &stdout, &stderr))
            }
            WaitOutcome::Cancelled => {
                kill_process_group(child_pid, &mut child).await;
                let grace = Some(Duration::from_millis(DRAIN_AFTER_KILL_MS));
                let stdout = collect_stream(stdout_task, grace).await;
                let stderr = collect_stream(stderr_task, grace).await;
                ToolOutput::error(format_output(
                    -1,
                    Some("command cancelled (process group terminated)"),
                    &stdout,
                    &stderr,
                ))
            }
        }
    }
}

enum WaitOutcome {
    Exited(std::io::Result<std::process::ExitStatus>),
    TimedOut,
    Cancelled,
}

async fn kill_process_group(pid: Option<u32>, child: &mut Child) {
    #[cfg(unix)]
    if let Some(raw_pid) = pid
        && let Ok(pgid) = i32::try_from(raw_pid)
    {
        use nix::sys::signal::{Signal, killpg};
        use nix::unistd::Pid;

        let group = Pid::from_raw(pgid);
        let _ = killpg(group, Signal::SIGTERM);

        let deadline = tokio::time::Instant::now() + Duration::from_millis(KILL_GRACE_MS);
        let mut child_reaped = false;

        while tokio::time::Instant::now() < deadline {
            if !child_reaped && matches!(child.try_wait(), Ok(Some(_))) {
                child_reaped = true;
            }
            if child_reaped && killpg(group, None).is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Escalate to SIGKILL for the entire group in case a descendant trapped SIGTERM.
        let _ = killpg(group, Signal::SIGKILL);
        if !child_reaped {
            let _ = child.kill().await;
            let _ = tokio::time::timeout(Duration::from_millis(500), child.wait()).await;
        }
        return;
    }

    let _ = child.kill().await;
    let _ = tokio::time::timeout(Duration::from_millis(500), child.wait()).await;
}

async fn collect_stream(
    handle: Option<tokio::task::JoinHandle<CapturedStream>>,
    timeout: Option<Duration>,
) -> CapturedStream {
    let Some(task) = handle else {
        return CapturedStream {
            text: String::new(),
            total_bytes: 0,
            truncated: false,
        };
    };

    let joined = match timeout {
        Some(limit) => match tokio::time::timeout(limit, task).await {
            Ok(Ok(stream)) => Some(stream),
            _ => None,
        },
        None => task.await.ok(),
    };

    joined.unwrap_or(CapturedStream {
        text: String::new(),
        total_bytes: 0,
        truncated: false,
    })
}

async fn drain_capped_stream<R: AsyncRead + Unpin>(
    mut reader: R,
    max_bytes: usize,
) -> CapturedStream {
    let head_budget = max_bytes / 2;
    let tail_budget = max_bytes.saturating_sub(head_budget);

    let mut head: Vec<u8> = Vec::new();
    let mut tail: VecDeque<u8> = VecDeque::new();
    let mut total_bytes = 0usize;
    let mut buf = [0u8; 8192];

    loop {
        match reader.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let slice = &buf[..n];
                total_bytes += n;

                if head.len() < head_budget {
                    let take = (head_budget - head.len()).min(n);
                    head.extend_from_slice(&slice[..take]);
                    if take < n {
                        push_tail(&mut tail, &slice[take..], tail_budget);
                    }
                } else {
                    push_tail(&mut tail, slice, tail_budget);
                }
            }
        }
    }

    if total_bytes <= max_bytes {
        head.extend(tail);
        return CapturedStream {
            text: String::from_utf8_lossy(&head).into_owned(),
            total_bytes,
            truncated: false,
        };
    }

    let tail_bytes: Vec<u8> = tail.into_iter().collect();
    let kept = head.len() + tail_bytes.len();
    let omitted = total_bytes.saturating_sub(kept);
    let head_str = String::from_utf8_lossy(&head);
    let tail_str = String::from_utf8_lossy(&tail_bytes);
    let text = format!(
        "{head_str}\n... [truncated {omitted} bytes of {total_bytes} total bytes] ...\n{tail_str}"
    );
    CapturedStream {
        text,
        total_bytes,
        truncated: true,
    }
}

fn push_tail(tail: &mut VecDeque<u8>, bytes: &[u8], cap: usize) {
    if cap == 0 {
        return;
    }
    if bytes.len() >= cap {
        tail.clear();
        tail.extend(&bytes[bytes.len() - cap..]);
        return;
    }
    let overflow = (tail.len() + bytes.len()).saturating_sub(cap);
    for _ in 0..overflow {
        tail.pop_front();
    }
    tail.extend(bytes);
}

fn format_output(
    exit_code: i32,
    note: Option<&str>,
    stdout: &CapturedStream,
    stderr: &CapturedStream,
) -> String {
    let mut out = String::new();
    if let Some(msg) = note {
        let _ = writeln!(out, "{msg}");
    }
    let _ = writeln!(out, "exit_code: {exit_code}");

    if !stdout.text.is_empty() {
        let _ = writeln!(
            out,
            "--- stdout ({} bytes{}) ---",
            stdout.total_bytes,
            if stdout.truncated { ", truncated" } else { "" }
        );
        out.push_str(&stdout.text);
        if !stdout.text.ends_with('\n') {
            out.push('\n');
        }
    }

    if !stderr.text.is_empty() {
        let _ = writeln!(
            out,
            "--- stderr ({} bytes{}) ---",
            stderr.total_bytes,
            if stderr.truncated { ", truncated" } else { "" }
        );
        out.push_str(&stderr.text);
        if !stderr.text.ends_with('\n') {
            out.push('\n');
        }
    }

    out.trim_end().to_owned()
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use tokio_util::sync::CancellationToken;

    use super::*;

    fn dummy_ctx() -> ToolContext {
        ToolContext {
            turn_id: "t1".into(),
            call_id: "c1".into(),
            cancel: CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn bash_executes_command_in_workspace_and_reports_exit_status() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("marker.txt"), "ok").expect("seed");
        let ws = WorkspaceContext::new(dir.path()).expect("ws");
        let tool = BashTool::new(ws);

        let ok = tool
            .execute(json!({ "command": "cat marker.txt" }), &dummy_ctx())
            .await;
        assert!(!ok.is_error, "unexpected error: {}", ok.content);
        assert!(ok.content.contains("exit_code: 0"));
        assert!(ok.content.contains("ok"));

        let fail = tool
            .execute(json!({ "command": "echo boom >&2; exit 7" }), &dummy_ctx())
            .await;
        assert!(fail.is_error);
        assert!(fail.content.contains("exit_code: 7"));
        assert!(fail.content.contains("boom"));
    }

    #[tokio::test]
    async fn bash_kills_entire_process_group_on_timeout_without_orphans() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ws = WorkspaceContext::new(dir.path()).expect("ws");
        let tool = BashTool::new(ws);

        let start = Instant::now();
        let out = tool
            .execute(
                json!({
                    "command": "(trap '' TERM; sleep 30) & echo $! > grandchild.pid; wait",
                    "timeout_secs": 1
                }),
                &dummy_ctx(),
            )
            .await;
        let elapsed = start.elapsed();

        assert!(out.is_error);
        assert!(out.content.contains("timed out"));
        assert!(
            elapsed < Duration::from_secs(4),
            "timeout cleanup took too long: {elapsed:?}"
        );

        #[cfg(unix)]
        {
            let pid_text = std::fs::read_to_string(dir.path().join("grandchild.pid"))
                .expect("grandchild pid file should exist");
            let grandchild_pid: i32 = pid_text.trim().parse().expect("valid pid");
            // Give the kernel a brief moment to finish terminating after SIGKILL.
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(
                !is_process_running(grandchild_pid),
                "grandchild process {grandchild_pid} leaked as an orphan!"
            );
        }
    }

    #[cfg(unix)]
    fn is_process_running(pid: i32) -> bool {
        use nix::sys::signal::kill;
        use nix::unistd::Pid;

        if kill(Pid::from_raw(pid), None).is_err() {
            return false;
        }
        if let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat"))
            && let Some((_, rest)) = stat.rsplit_once(") ")
            && rest.starts_with('Z')
        {
            return false;
        }
        true
    }

    #[tokio::test]
    async fn bash_truncates_large_output_preserving_head_and_tail() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ws = WorkspaceContext::new(dir.path()).expect("ws");
        let tool = BashTool::new(ws).with_max_output_bytes(512);

        let out = tool
            .execute(
                json!({
                    "command": "echo START_MARKER; for i in $(seq 1 200); do echo line_$i; done; echo END_MARKER"
                }),
                &dummy_ctx(),
            )
            .await;
        assert!(!out.is_error, "unexpected error: {}", out.content);
        assert!(out.content.contains("START_MARKER"));
        assert!(out.content.contains("END_MARKER"));
        assert!(out.content.contains("truncated"));
    }
}
