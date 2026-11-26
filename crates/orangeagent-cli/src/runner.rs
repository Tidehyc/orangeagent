use std::io::Write;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use orangeagent_ai::{
    LlmProvider, Message, OpenAiChatConfig, OpenAiChatProvider, ProviderFamily, Role, VcrProvider,
};
use orangeagent_core::{
    AgentLoop, CompositeHook, LoopConfig, SteerQueue, ToolRegistry, TurnOutcome,
};
use orangeagent_harness::{WorkspaceContext, register_core_tools};
use orangeagent_spine::JsonlSessionStore;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::cli::RunArgs;
use crate::render::{OutputMode, render_event_stream};

/// Static, cache-friendly default system prompt.
///
/// Intentionally contains zero dynamic timestamps, random IDs, or per-turn counters so
/// provider prompt caches stay warm across multi-turn coding sessions.
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are OrangeAgent, a precise pure-Rust coding agent operating inside a local workspace.\n\
Follow these rules strictly:\n\
1. Always inspect relevant files with `read`, `grep`, or `glob` before modifying them.\n\
2. Prefer minimal, surgical `edit` hunks over rewriting whole files with `write`.\n\
3. Verify code changes with `bash` (build/test/lint) before concluding.\n\
4. Keep final responses concise and grounded in observed tool outputs.";

/// Structured result returned by [`execute_run`].
#[derive(Debug)]
pub struct RunReport {
    pub outcome: TurnOutcome,
    pub persisted_entries: usize,
}

/// Execute a single `orangeagent run` turn end-to-end across all workspace crates.
pub async fn execute_run<WOut, WErr>(
    args: &RunArgs,
    stdout: WOut,
    stderr: WErr,
    cancel: CancellationToken,
) -> Result<RunReport, String>
where
    WOut: Write + Send + 'static,
    WErr: Write + Send + 'static,
{
    let workspace = WorkspaceContext::new(&args.workspace)
        .map_err(|err| format!("invalid workspace `{}`: {err}", args.workspace.display()))?;

    let provider = build_provider(args)?;

    let mut registry = ToolRegistry::new();
    register_core_tools(&mut registry, workspace.clone());

    let mut session_store = match &args.session {
        Some(path) => {
            let session_id = default_session_id(path);
            let root_str = workspace.root().display().to_string();
            let store = JsonlSessionStore::open_or_create(path, session_id, root_str, None)
                .map_err(|err| format!("failed to open session `{}`: {err}", path.display()))?;
            Some(store)
        }
        None => None,
    };

    let mut messages: Vec<Message> = match &session_store {
        Some(store) => store
            .branch_messages(None)
            .map_err(|err| format!("failed to load session history: {err}"))?,
        None => Vec::new(),
    };

    let initial_len = messages.len();
    messages.push(Message::user(&args.prompt));

    let mut config = LoopConfig::new(&args.model);
    config.system = Some(
        args.system
            .clone()
            .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_owned()),
    );
    config.max_iterations = args.max_iterations.max(1);

    let agent = AgentLoop::new(provider, registry, CompositeHook::new(), config);
    let steer = SteerQueue::new();

    let mode = if args.json {
        OutputMode::JsonLines
    } else {
        OutputMode::Human
    };
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let render_handle = tokio::spawn(render_event_stream(event_rx, mode, stdout, stderr));

    let turn_id = make_turn_id();
    let outcome = agent
        .run_turn(turn_id, &mut messages, &steer, Some(&event_tx), cancel)
        .await;

    // Close channel so renderer flushes and exits cleanly.
    drop(event_tx);
    match render_handle.await {
        Ok(Ok(())) => {}
        Ok(Err(err)) => return Err(format!("output rendering failed: {err}")),
        Err(join_err) => return Err(format!("render task panicked: {join_err}")),
    }

    let mut persisted_entries = 0usize;
    if let Some(store) = session_store.as_mut() {
        let final_usage = outcome_usage(&outcome);
        let new_slice = &messages[initial_len..];
        let last_assistant_idx = new_slice
            .iter()
            .rposition(|msg| msg.role == Role::Assistant);

        for (idx, msg) in new_slice.iter().enumerate() {
            let model = (msg.role == Role::Assistant).then(|| args.model.clone());
            let usage = (Some(idx) == last_assistant_idx).then_some(final_usage);
            store
                .append_message(msg.clone(), model, usage, None)
                .map_err(|err| format!("failed to persist session entry: {err}"))?;
            persisted_entries += 1;
        }
    }

    Ok(RunReport {
        outcome,
        persisted_entries,
    })
}

/// Map a [`TurnOutcome`] to a standard Unix process exit code.
pub fn exit_code_for_outcome(outcome: &TurnOutcome) -> ExitCode {
    match outcome {
        TurnOutcome::Completed { .. } => ExitCode::SUCCESS,
        TurnOutcome::Interrupted { .. } => ExitCode::from(130),
        TurnOutcome::Failed { .. } => ExitCode::FAILURE,
    }
}

fn build_provider(args: &RunArgs) -> Result<Arc<dyn LlmProvider>, String> {
    if let Some(cassette_path) = &args.vcr_replay {
        let provider = VcrProvider::replay_file(
            ProviderFamily::OpenAiChat,
            cassette_path,
            args.vcr_strict_hash,
        )
        .map_err(|err| {
            format!(
                "failed to load VCR cassette `{}`: {err}",
                cassette_path.display()
            )
        })?;
        return Ok(Arc::new(provider));
    }

    let api_key = args
        .api_key
        .clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .filter(|k| !k.trim().is_empty());

    let live = OpenAiChatProvider::new(OpenAiChatConfig::new(&args.base_url, api_key))
        .map_err(|err| format!("failed to initialize OpenAI Chat provider: {err}"))?;
    let live_arc: Arc<dyn LlmProvider> = Arc::new(live);

    if let Some(record_path) = &args.vcr_record {
        Ok(Arc::new(VcrProvider::record(live_arc, record_path)))
    } else {
        Ok(live_arc)
    }
}

fn outcome_usage(outcome: &TurnOutcome) -> orangeagent_ai::TokenUsage {
    match outcome {
        TurnOutcome::Completed { total_usage, .. }
        | TurnOutcome::Interrupted { total_usage, .. }
        | TurnOutcome::Failed { total_usage, .. } => *total_usage,
    }
}

fn default_session_id(path: &std::path::Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .map_or_else(make_turn_id, ToOwned::to_owned)
}

fn make_turn_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    format!("turn-{millis}")
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use orangeagent_ai::{
        FinishReason, StreamEvent, TokenUsage, VcrCassette, VcrInteraction, VcrItem,
    };
    use orangeagent_core::AgentEvent;
    use serde_json::json;

    use super::*;

    #[derive(Clone, Default)]
    struct SharedBuf(Arc<Mutex<Vec<u8>>>);

    impl SharedBuf {
        fn as_text(&self) -> String {
            let guard = self
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            String::from_utf8_lossy(&guard).into_owned()
        }
    }

    impl Write for SharedBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let mut guard = self
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn base_args(workspace: &std::path::Path, cassette: &std::path::Path) -> RunArgs {
        RunArgs {
            prompt: "fix the bug in calc.rs".into(),
            workspace: workspace.to_path_buf(),
            model: "test-model".into(),
            base_url: "http://localhost:0/v1".into(),
            api_key: None,
            system: None,
            max_iterations: 20,
            session: None,
            json: true,
            vcr_replay: Some(cassette.to_path_buf()),
            vcr_record: None,
            vcr_strict_hash: false,
        }
    }

    fn interaction_with_events(events: Vec<StreamEvent>) -> VcrInteraction {
        VcrInteraction {
            request_hash: "ignored-in-relaxed-mode".into(),
            model: "test-model".into(),
            initial_error: None,
            items: events
                .into_iter()
                .map(|event| VcrItem::Event { event })
                .collect(),
        }
    }

    #[tokio::test]
    async fn e2e_vcr_multi_tool_read_edit_bash_and_session_persistence() {
        let dir = tempfile::tempdir().expect("tempdir");
        let calc_path = dir.path().join("calc.rs");
        std::fs::write(
            &calc_path,
            "pub fn add(a: i32, b: i32) -> i32 {\n    a - b\n}\n",
        )
        .expect("seed calc.rs");

        let cassette_path = dir.path().join("cassette.jsonl");
        let session_path = dir.path().join("session.jsonl");

        let cassette = VcrCassette {
            interactions: vec![
                // Iteration 1: parallel read + glob
                interaction_with_events(vec![
                    StreamEvent::ToolCallStart {
                        index: 0,
                        id: "call_read".into(),
                        name: "read".into(),
                    },
                    StreamEvent::ToolCallArgsDelta {
                        index: 0,
                        json_fragment: json!({ "path": "calc.rs" }).to_string(),
                    },
                    StreamEvent::ToolCallStart {
                        index: 1,
                        id: "call_glob".into(),
                        name: "glob".into(),
                    },
                    StreamEvent::ToolCallArgsDelta {
                        index: 1,
                        json_fragment: json!({ "pattern": "**/*.rs" }).to_string(),
                    },
                    StreamEvent::Usage(TokenUsage {
                        input_tokens: 500,
                        output_tokens: 40,
                        cache_read_tokens: 0,
                        cache_write_tokens: 500,
                        reasoning_tokens: 0,
                    }),
                    StreamEvent::Done {
                        reason: FinishReason::ToolCalls,
                    },
                ]),
                // Iteration 2: edit calc.rs
                interaction_with_events(vec![
                    StreamEvent::ToolCallStart {
                        index: 0,
                        id: "call_edit".into(),
                        name: "edit".into(),
                    },
                    StreamEvent::ToolCallArgsDelta {
                        index: 0,
                        json_fragment: json!({
                            "path": "calc.rs",
                            "old_text": "    a - b",
                            "new_text": "    a + b"
                        })
                        .to_string(),
                    },
                    StreamEvent::Usage(TokenUsage {
                        input_tokens: 700,
                        output_tokens: 50,
                        cache_read_tokens: 520,
                        cache_write_tokens: 0,
                        reasoning_tokens: 0,
                    }),
                    StreamEvent::Done {
                        reason: FinishReason::ToolCalls,
                    },
                ]),
                // Iteration 3: verify with bash
                interaction_with_events(vec![
                    StreamEvent::ToolCallStart {
                        index: 0,
                        id: "call_bash".into(),
                        name: "bash".into(),
                    },
                    StreamEvent::ToolCallArgsDelta {
                        index: 0,
                        json_fragment: json!({
                            "command": "grep -q 'a + b' calc.rs && echo VERIFIED"
                        })
                        .to_string(),
                    },
                    StreamEvent::Usage(TokenUsage {
                        input_tokens: 850,
                        output_tokens: 35,
                        cache_read_tokens: 710,
                        cache_write_tokens: 0,
                        reasoning_tokens: 0,
                    }),
                    StreamEvent::Done {
                        reason: FinishReason::ToolCalls,
                    },
                ]),
                // Iteration 4: final answer
                interaction_with_events(vec![
                    StreamEvent::TextDelta {
                        text: "Fixed `calc.rs` and verified.".into(),
                    },
                    StreamEvent::Usage(TokenUsage {
                        input_tokens: 920,
                        output_tokens: 20,
                        cache_read_tokens: 850,
                        cache_write_tokens: 0,
                        reasoning_tokens: 0,
                    }),
                    StreamEvent::Done {
                        reason: FinishReason::Stop,
                    },
                ]),
            ],
        };
        cassette.save_jsonl(&cassette_path).expect("save cassette");

        let mut args = base_args(dir.path(), &cassette_path);
        args.session = Some(session_path.clone());

        let stdout = SharedBuf::default();
        let stderr = SharedBuf::default();
        let report = execute_run(
            &args,
            stdout.clone(),
            stderr.clone(),
            CancellationToken::new(),
        )
        .await
        .expect("run should succeed");

        assert!(matches!(
            report.outcome,
            TurnOutcome::Completed { iterations: 4, .. }
        ));
        assert_eq!(exit_code_for_outcome(&report.outcome), ExitCode::SUCCESS);

        // Disk file was genuinely fixed by EditTool.
        let disk = std::fs::read_to_string(&calc_path).expect("read calc.rs");
        assert!(disk.contains("a + b"));
        assert!(!disk.contains("a - b"));

        // Every stdout line parses as a typed AgentEvent in `--json` mode.
        let raw_stdout = stdout.as_text();
        let events: Vec<AgentEvent> = raw_stdout
            .lines()
            .map(|line| serde_json::from_str(line).expect("valid AgentEvent JSONL"))
            .collect();
        assert!(matches!(
            events.first(),
            Some(AgentEvent::TurnStarted { .. })
        ));
        assert!(matches!(
            events.last(),
            Some(AgentEvent::TurnFinished { .. })
        ));

        let finished_tools: Vec<(&str, bool)> = events
            .iter()
            .filter_map(|ev| match ev {
                AgentEvent::ToolCallFinished { name, is_error, .. } => {
                    Some((name.as_str(), *is_error))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            finished_tools,
            vec![
                ("read", false),
                ("glob", false),
                ("edit", false),
                ("bash", false),
            ]
        );

        // Session JSONL persisted all turn messages and reloads cleanly.
        assert!(report.persisted_entries >= 8);
        let reopened = JsonlSessionStore::open(&session_path).expect("reopen session");
        let persisted_msgs = reopened.branch_messages(None).expect("branch msgs");
        assert_eq!(persisted_msgs.len(), report.persisted_entries);
        assert_eq!(persisted_msgs[0].text_content(), "fix the bug in calc.rs");
        assert_eq!(
            persisted_msgs.last().expect("last").text_content(),
            "Fixed `calc.rs` and verified."
        );
    }

    #[tokio::test]
    async fn e2e_vcr_ten_round_prompt_cache_hit_ratio_exceeds_80_percent() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("README.md"), "# orangeagent\n").expect("seed");
        let cassette_path = dir.path().join("cache10.jsonl");

        let mut interactions = Vec::new();
        let mut cumulative_input = 1000u64;
        for round in 1..=9 {
            let cached = if round == 1 {
                0
            } else {
                cumulative_input - 120
            };
            interactions.push(interaction_with_events(vec![
                StreamEvent::ToolCallStart {
                    index: 0,
                    id: format!("call_{round}"),
                    name: "read".into(),
                },
                StreamEvent::ToolCallArgsDelta {
                    index: 0,
                    json_fragment: json!({ "path": "README.md" }).to_string(),
                },
                StreamEvent::Usage(TokenUsage {
                    input_tokens: cumulative_input,
                    output_tokens: 30,
                    cache_read_tokens: cached,
                    cache_write_tokens: if round == 1 { cumulative_input } else { 0 },
                    reasoning_tokens: 0,
                }),
                StreamEvent::Done {
                    reason: FinishReason::ToolCalls,
                },
            ]));
            cumulative_input += 120;
        }
        // Round 10 final completion.
        interactions.push(interaction_with_events(vec![
            StreamEvent::TextDelta {
                text: "Completed 10-round cache benchmark.".into(),
            },
            StreamEvent::Usage(TokenUsage {
                input_tokens: cumulative_input,
                output_tokens: 25,
                cache_read_tokens: cumulative_input - 100,
                cache_write_tokens: 0,
                reasoning_tokens: 0,
            }),
            StreamEvent::Done {
                reason: FinishReason::Stop,
            },
        ]));

        VcrCassette { interactions }
            .save_jsonl(&cassette_path)
            .expect("save cache cassette");

        let mut args = base_args(dir.path(), &cassette_path);
        args.json = false; // also exercise Human rendering mode + cache ratio footer

        let stdout = SharedBuf::default();
        let stderr = SharedBuf::default();
        let report = execute_run(
            &args,
            stdout.clone(),
            stderr.clone(),
            CancellationToken::new(),
        )
        .await
        .expect("10-round run");

        match report.outcome {
            TurnOutcome::Completed {
                iterations,
                total_usage,
                ..
            } => {
                assert_eq!(iterations, 10);
                assert!(
                    total_usage.cache_hit_ratio() > 0.80,
                    "expected >80% cache hit ratio, got {:.2}%",
                    total_usage.cache_hit_ratio() * 100.0
                );
            }
            other => panic!("expected Completed, got {other:?}"),
        }

        assert!(
            stdout
                .as_text()
                .contains("Completed 10-round cache benchmark.")
        );
        assert!(stderr.as_text().contains("[done] iterations=10"));
    }

    #[tokio::test]
    async fn e2e_session_resume_loads_prior_branch_and_appends_next_turn() {
        let dir = tempfile::tempdir().expect("tempdir");
        let session_path = dir.path().join("resume.jsonl");
        let c1_path = dir.path().join("turn1.jsonl");
        let c2_path = dir.path().join("turn2.jsonl");

        VcrCassette {
            interactions: vec![interaction_with_events(vec![
                StreamEvent::TextDelta {
                    text: "Turn 1 answer.".into(),
                },
                StreamEvent::Done {
                    reason: FinishReason::Stop,
                },
            ])],
        }
        .save_jsonl(&c1_path)
        .expect("save c1");

        VcrCassette {
            interactions: vec![interaction_with_events(vec![
                StreamEvent::TextDelta {
                    text: "Turn 2 follow-up answer.".into(),
                },
                StreamEvent::Done {
                    reason: FinishReason::Stop,
                },
            ])],
        }
        .save_jsonl(&c2_path)
        .expect("save c2");

        let mut args1 = base_args(dir.path(), &c1_path);
        args1.prompt = "First question".into();
        args1.session = Some(session_path.clone());
        execute_run(
            &args1,
            SharedBuf::default(),
            SharedBuf::default(),
            CancellationToken::new(),
        )
        .await
        .expect("turn 1");

        let mut args2 = base_args(dir.path(), &c2_path);
        args2.prompt = "Second question".into();
        args2.session = Some(session_path.clone());
        execute_run(
            &args2,
            SharedBuf::default(),
            SharedBuf::default(),
            CancellationToken::new(),
        )
        .await
        .expect("turn 2");

        let reopened = JsonlSessionStore::open(&session_path).expect("open resumed session");
        let texts: Vec<String> = reopened
            .branch_messages(None)
            .expect("msgs")
            .iter()
            .map(Message::text_content)
            .collect();
        assert_eq!(
            texts,
            vec![
                "First question",
                "Turn 1 answer.",
                "Second question",
                "Turn 2 follow-up answer.",
            ]
        );
    }
}
