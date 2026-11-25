use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use futures::future::join_all;
use orangeagent_ai::{
    AiError, ContentBlock, FinishReason, LlmProvider, Message, ProviderRequest, StreamEvent,
    TokenUsage, sanitize_for_provider,
};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::{
    AgentEvent, CompositeHook, ConcurrencyMode, EventSink, HookDecision, HookEvent, HookPhase,
    RecoveryConfig, RecoveryState, Tool, ToolContext, ToolOutput, ToolRegistry, TurnOutcome,
    parse_tool_arguments,
};

/// Thread-safe FIFO mailbox for injecting mid-turn user steering messages.
///
/// Uses a short-lived `std::sync::Mutex` (never held across `.await`) so external
/// schedulers (e.g. `orangeagent-spine` `BusyPolicy::Inject`) can push messages
/// synchronously from any thread without async lock overhead.
#[derive(Debug, Default, Clone)]
pub struct SteerQueue {
    inner: Arc<Mutex<VecDeque<String>>>,
}

impl SteerQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue a steering message to be injected before the next LLM call.
    pub fn push(&self, message: impl Into<String>) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.push_back(message.into());
    }

    /// Drain all pending steering messages in FIFO order.
    pub fn drain_all(&self) -> Vec<String> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.drain(..).collect()
    }

    pub fn is_empty(&self) -> bool {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.is_empty()
    }
}

/// Runtime parameters governing a single [`AgentLoop::run_turn`] execution.
#[derive(Debug, Clone)]
pub struct LoopConfig {
    pub model: String,
    pub system: Option<String>,
    pub max_iterations: usize,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    /// Maximum Unicode scalar values included in [`AgentEvent::ToolCallFinished::output_preview`].
    pub output_preview_chars: usize,
    pub recovery: RecoveryConfig,
}

impl LoopConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            system: None,
            max_iterations: 25,
            max_tokens: None,
            temperature: None,
            output_preview_chars: 200,
            recovery: RecoveryConfig::default(),
        }
    }
}

/// Minimal event-driven ReAct loop.
///
/// Responsibilities:
/// - Stream LLM deltas into structured assistant messages + [`AgentEvent`]s.
/// - Dispatch all 6 [`HookPhase`] checkpoints via [`CompositeHook`].
/// - Batch contiguous [`ConcurrencyMode::ReadOnlyParallel`] tool calls via `join_all`
///   while executing [`ConcurrencyMode::ExclusiveSequential`] calls strictly in order.
/// - Self-heal empty responses and malformed tool JSON within bounded budgets.
/// - Drain [`SteerQueue`] after tool execution and honor [`CancellationToken`] promptly.
pub struct AgentLoop {
    provider: Arc<dyn LlmProvider>,
    registry: ToolRegistry,
    hooks: CompositeHook,
    config: LoopConfig,
}

impl AgentLoop {
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        registry: ToolRegistry,
        hooks: CompositeHook,
        config: LoopConfig,
    ) -> Self {
        Self {
            provider,
            registry,
            hooks,
            config,
        }
    }

    /// Execute one complete user turn, mutating `messages` in place with assistant
    /// responses, tool results, and any drained mid-turn steering inputs.
    pub async fn run_turn(
        &self,
        turn_id: impl Into<String>,
        messages: &mut Vec<Message>,
        steer: &SteerQueue,
        events: Option<&EventSink>,
        cancel: CancellationToken,
    ) -> TurnOutcome {
        let turn_id = turn_id.into();
        let mut total_usage = TokenUsage::default();
        let mut iterations = 0usize;
        let mut recovery = RecoveryState::new(self.config.recovery);

        emit(
            events,
            AgentEvent::TurnStarted {
                turn_id: turn_id.clone(),
            },
        );

        let before_turn = self
            .hooks
            .dispatch(&HookEvent {
                phase: HookPhase::BeforeTurn,
                turn_id: turn_id.clone(),
                tool_name: None,
                payload: json!({ "message_count": messages.len() }),
            })
            .await;

        if let Some(early) = handle_turn_level_decision(before_turn, iterations, total_usage) {
            return self.finish_turn(&turn_id, early, events).await;
        }

        while iterations < self.config.max_iterations {
            if cancel.is_cancelled() {
                return self
                    .finish_turn(
                        &turn_id,
                        TurnOutcome::Interrupted {
                            reason: "cancelled before model call".into(),
                            iterations,
                            total_usage,
                        },
                        events,
                    )
                    .await;
            }

            iterations += 1;
            emit(
                events,
                AgentEvent::IterationStarted {
                    iteration: iterations,
                },
            );

            if let Some(outcome) = self
                .run_iteration(
                    &turn_id,
                    iterations,
                    messages,
                    steer,
                    &mut recovery,
                    &mut total_usage,
                    events,
                    &cancel,
                )
                .await
            {
                return self.finish_turn(&turn_id, outcome, events).await;
            }
        }

        self.finish_turn(
            &turn_id,
            TurnOutcome::Failed {
                error: format!(
                    "exceeded maximum iteration budget ({})",
                    self.config.max_iterations
                ),
                iterations,
                total_usage,
            },
            events,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_iteration(
        &self,
        turn_id: &str,
        iterations: usize,
        messages: &mut Vec<Message>,
        steer: &SteerQueue,
        recovery: &mut RecoveryState,
        total_usage: &mut TokenUsage,
        events: Option<&EventSink>,
        cancel: &CancellationToken,
    ) -> Option<TurnOutcome> {
        let model_turn = match self
            .invoke_model(turn_id, iterations, messages, total_usage, events, cancel)
            .await
        {
            Ok(turn) => turn,
            Err(outcome) => return Some(outcome),
        };

        if model_turn.tool_calls.is_empty() {
            return self.handle_no_tool_turn(
                model_turn,
                messages,
                recovery,
                events,
                iterations,
                *total_usage,
            );
        }

        let prepared = match prepare_tool_calls(model_turn.tool_calls, recovery, events) {
            Ok(calls) => calls,
            Err(error) => {
                return Some(TurnOutcome::Failed {
                    error,
                    iterations,
                    total_usage: *total_usage,
                });
            }
        };

        messages.push(build_tool_call_assistant_message(
            model_turn.thinking,
            model_turn.thinking_signature,
            self.provider.family(),
            model_turn.text,
            &prepared,
        ));

        if let Some(early) = self
            .execute_prepared_calls(
                turn_id,
                prepared,
                messages,
                events,
                cancel,
                iterations,
                *total_usage,
            )
            .await
        {
            return Some(early);
        }

        for steered in steer.drain_all() {
            messages.push(Message::user(steered));
        }

        None
    }

    async fn invoke_model(
        &self,
        turn_id: &str,
        iterations: usize,
        messages: &[Message],
        total_usage: &mut TokenUsage,
        events: Option<&EventSink>,
        cancel: &CancellationToken,
    ) -> Result<ModelTurn, TurnOutcome> {
        let before_model = self
            .hooks
            .dispatch(&HookEvent {
                phase: HookPhase::BeforeModelCall,
                turn_id: turn_id.to_string(),
                tool_name: None,
                payload: json!({ "iteration": iterations, "model": self.config.model }),
            })
            .await;

        if let Some(early) = handle_turn_level_decision(before_model, iterations, *total_usage) {
            return Err(early);
        }

        let req = ProviderRequest {
            model: self.config.model.clone(),
            system: self.config.system.clone(),
            messages: sanitize_for_provider(messages, self.provider.family()),
            tools: self.registry.definitions(),
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
        };

        let model_turn = match self
            .collect_model_stream(req, total_usage, events, cancel.clone())
            .await
        {
            Ok(turn) => turn,
            Err(AiError::Cancelled) => {
                return Err(TurnOutcome::Interrupted {
                    reason: "cancelled during model stream".into(),
                    iterations,
                    total_usage: *total_usage,
                });
            }
            Err(err) => {
                return Err(TurnOutcome::Failed {
                    error: err.to_string(),
                    iterations,
                    total_usage: *total_usage,
                });
            }
        };

        let after_model = self
            .hooks
            .dispatch(&HookEvent {
                phase: HookPhase::AfterModelCall,
                turn_id: turn_id.to_string(),
                tool_name: None,
                payload: json!({
                    "iteration": iterations,
                    "text_len": model_turn.text.len(),
                    "tool_call_count": model_turn.tool_calls.len(),
                    "finish_reason": model_turn.finish_reason,
                }),
            })
            .await;

        if let Some(early) = handle_turn_level_decision(after_model, iterations, *total_usage) {
            return Err(early);
        }

        Ok(model_turn)
    }

    fn handle_no_tool_turn(
        &self,
        model_turn: ModelTurn,
        messages: &mut Vec<Message>,
        recovery: &mut RecoveryState,
        events: Option<&EventSink>,
        iterations: usize,
        total_usage: TokenUsage,
    ) -> Option<TurnOutcome> {
        if model_turn.text.trim().is_empty() {
            if let Some((attempt, prompt)) = recovery.on_empty_response() {
                emit(
                    events,
                    AgentEvent::RecoveryTriggered {
                        attempt,
                        reason: "empty assistant response".into(),
                    },
                );
                messages.push(Message::user(prompt));
                return None;
            }
            return Some(TurnOutcome::Failed {
                error: "model produced consecutive empty responses".into(),
                iterations,
                total_usage,
            });
        }

        recovery.reset();
        messages.push(model_turn.into_text_assistant_message(self.provider.family()));
        let final_text = messages.last().map_or_default(Message::text_content);
        Some(TurnOutcome::Completed {
            final_text,
            iterations,
            total_usage,
        })
    }

    async fn collect_model_stream(
        &self,
        req: ProviderRequest,
        total_usage: &mut TokenUsage,
        events: Option<&EventSink>,
        cancel: CancellationToken,
    ) -> Result<ModelTurn, AiError> {
        let mut stream = self.provider.stream(req, cancel).await?;
        let mut thinking = String::new();
        let mut thinking_signature: Option<String> = None;
        let mut text = String::new();
        let mut pending_calls: BTreeMap<usize, RawToolCall> = BTreeMap::new();
        let mut finish_reason: Option<FinishReason> = None;

        while let Some(item) = stream.next().await {
            match item? {
                StreamEvent::ThinkingDelta { text: delta } => {
                    thinking.push_str(&delta);
                    emit(events, AgentEvent::ThinkingDelta { text: delta });
                }
                StreamEvent::ThinkingSignature { signature } => {
                    thinking_signature = Some(signature);
                }
                StreamEvent::TextDelta { text: delta } => {
                    text.push_str(&delta);
                    emit(events, AgentEvent::TextDelta { text: delta });
                }
                StreamEvent::ToolCallStart { index, id, name } => {
                    pending_calls.insert(
                        index,
                        RawToolCall {
                            id,
                            name,
                            raw_args: String::new(),
                        },
                    );
                }
                StreamEvent::ToolCallArgsDelta {
                    index,
                    json_fragment,
                } => {
                    if let Some(call) = pending_calls.get_mut(&index) {
                        call.raw_args.push_str(&json_fragment);
                    }
                }
                StreamEvent::Usage(usage) => {
                    total_usage.accumulate(&usage);
                    emit(events, AgentEvent::UsageUpdated { usage });
                }
                StreamEvent::Done { reason } => {
                    finish_reason = Some(reason);
                }
            }
        }

        Ok(ModelTurn {
            thinking,
            thinking_signature,
            text,
            tool_calls: pending_calls.into_values().collect(),
            finish_reason,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_prepared_calls(
        &self,
        turn_id: &str,
        calls: Vec<PreparedToolCall>,
        messages: &mut Vec<Message>,
        events: Option<&EventSink>,
        cancel: &CancellationToken,
        iterations: usize,
        total_usage: TokenUsage,
    ) -> Option<TurnOutcome> {
        for batch in partition_tool_batches(calls, &self.registry) {
            if cancel.is_cancelled() {
                return Some(TurnOutcome::Interrupted {
                    reason: "cancelled before tool batch".into(),
                    iterations,
                    total_usage,
                });
            }

            match batch {
                ToolBatch::Parallel(group) => {
                    let futures = group
                        .into_iter()
                        .map(|call| self.run_single_tool(turn_id, call, events, cancel));
                    let results = join_all(futures).await;
                    for result in results {
                        match result {
                            SingleToolOutcome::Completed { call_id, output } => {
                                messages.push(Message::tool_result(
                                    call_id,
                                    output.content,
                                    output.is_error,
                                ));
                            }
                            SingleToolOutcome::ShortCircuit(final_text) => {
                                return Some(TurnOutcome::Interrupted {
                                    reason: final_text,
                                    iterations,
                                    total_usage,
                                });
                            }
                        }
                    }
                }
                ToolBatch::Sequential(call) => {
                    match self.run_single_tool(turn_id, call, events, cancel).await {
                        SingleToolOutcome::Completed { call_id, output } => {
                            messages.push(Message::tool_result(
                                call_id,
                                output.content,
                                output.is_error,
                            ));
                        }
                        SingleToolOutcome::ShortCircuit(final_text) => {
                            return Some(TurnOutcome::Interrupted {
                                reason: final_text,
                                iterations,
                                total_usage,
                            });
                        }
                    }
                }
            }
        }

        None
    }

    async fn run_single_tool(
        &self,
        turn_id: &str,
        call: PreparedToolCall,
        events: Option<&EventSink>,
        cancel: &CancellationToken,
    ) -> SingleToolOutcome {
        if let Some(pre_error) = call.malformed_error {
            let output = ToolOutput::error(pre_error);
            return self.complete_tool_call(call.id, call.name, output, events);
        }

        let mut args = call.arguments;
        let before = self
            .hooks
            .dispatch(&HookEvent {
                phase: HookPhase::BeforeToolUse,
                turn_id: turn_id.to_string(),
                tool_name: Some(call.name.clone()),
                payload: args.clone(),
            })
            .await;

        match before {
            HookDecision::PassThrough => {}
            HookDecision::Modify(updated_args) => args = updated_args,
            HookDecision::Deny(reason) => {
                let output = ToolOutput::error(format!("Tool call denied: {reason}"));
                return self.complete_tool_call(call.id, call.name, output, events);
            }
            HookDecision::ShortCircuit(text) => return SingleToolOutcome::ShortCircuit(text),
        }

        emit(
            events,
            AgentEvent::ToolCallStarted {
                call_id: call.id.clone(),
                name: call.name.clone(),
                arguments: args.clone(),
            },
        );

        let tool: Option<Arc<dyn Tool>> = self.registry.get(&call.name);
        let mut output = match tool {
            Some(tool) => {
                let ctx = ToolContext {
                    turn_id: turn_id.to_string(),
                    call_id: call.id.clone(),
                    cancel: cancel.clone(),
                };
                tool.execute(args, &ctx).await
            }
            None => ToolOutput::error(format!("Unknown tool `{}`", call.name)),
        };

        let after = self
            .hooks
            .dispatch(&HookEvent {
                phase: HookPhase::AfterToolUse,
                turn_id: turn_id.to_string(),
                tool_name: Some(call.name.clone()),
                payload: json!({
                    "call_id": call.id,
                    "content": output.content,
                    "is_error": output.is_error,
                }),
            })
            .await;

        match after {
            HookDecision::PassThrough => {}
            HookDecision::Modify(updated) => {
                if let Some(content) = updated.get("content").and_then(Value::as_str) {
                    output.content = content.to_string();
                }
                if let Some(is_error) = updated.get("is_error").and_then(Value::as_bool) {
                    output.is_error = is_error;
                }
            }
            HookDecision::Deny(reason) => {
                output = ToolOutput::error(format!("Tool output rejected by hook: {reason}"));
            }
            HookDecision::ShortCircuit(text) => return SingleToolOutcome::ShortCircuit(text),
        }

        self.complete_tool_call(call.id, call.name, output, events)
    }

    fn complete_tool_call(
        &self,
        call_id: String,
        name: String,
        output: ToolOutput,
        events: Option<&EventSink>,
    ) -> SingleToolOutcome {
        emit(
            events,
            AgentEvent::ToolCallFinished {
                call_id: call_id.clone(),
                name,
                is_error: output.is_error,
                output_preview: truncate_preview(&output.content, self.config.output_preview_chars),
            },
        );
        SingleToolOutcome::Completed { call_id, output }
    }

    async fn finish_turn(
        &self,
        turn_id: &str,
        outcome: TurnOutcome,
        events: Option<&EventSink>,
    ) -> TurnOutcome {
        let _ = self
            .hooks
            .dispatch(&HookEvent {
                phase: HookPhase::AfterTurn,
                turn_id: turn_id.to_string(),
                tool_name: None,
                payload: serde_json::to_value(&outcome).unwrap_or(Value::Null),
            })
            .await;

        emit(
            events,
            AgentEvent::TurnFinished {
                outcome: outcome.clone(),
            },
        );
        outcome
    }
}

struct RawToolCall {
    id: String,
    name: String,
    raw_args: String,
}

struct ModelTurn {
    thinking: String,
    thinking_signature: Option<String>,
    text: String,
    tool_calls: Vec<RawToolCall>,
    finish_reason: Option<FinishReason>,
}

impl ModelTurn {
    fn into_text_assistant_message(self, family: orangeagent_ai::ProviderFamily) -> Message {
        let mut content = Vec::new();
        if !self.thinking.is_empty() || self.thinking_signature.is_some() {
            content.push(ContentBlock::Thinking {
                text: self.thinking,
                signature: self.thinking_signature,
                provider: Some(family),
            });
        }
        if !self.text.is_empty() {
            content.push(ContentBlock::Text { text: self.text });
        }
        Message {
            role: orangeagent_ai::Role::Assistant,
            content,
        }
    }
}

struct PreparedToolCall {
    id: String,
    name: String,
    arguments: Value,
    malformed_error: Option<String>,
}

enum ToolBatch {
    Parallel(Vec<PreparedToolCall>),
    Sequential(PreparedToolCall),
}

enum SingleToolOutcome {
    Completed { call_id: String, output: ToolOutput },
    ShortCircuit(String),
}

fn prepare_tool_calls(
    raw_calls: Vec<RawToolCall>,
    recovery: &mut RecoveryState,
    events: Option<&EventSink>,
) -> Result<Vec<PreparedToolCall>, String> {
    let mut prepared = Vec::with_capacity(raw_calls.len());
    let mut had_valid_call = false;

    for raw in raw_calls {
        match parse_tool_arguments(&raw.raw_args) {
            Ok(arguments) => {
                had_valid_call = true;
                prepared.push(PreparedToolCall {
                    id: raw.id,
                    name: raw.name,
                    arguments,
                    malformed_error: None,
                });
            }
            Err(parse_err) => {
                if let Some((attempt, feedback)) =
                    recovery.on_malformed_tool_args(&raw.name, &parse_err)
                {
                    emit(
                        events,
                        AgentEvent::RecoveryTriggered {
                            attempt,
                            reason: format!("malformed JSON for tool `{}`", raw.name),
                        },
                    );
                    prepared.push(PreparedToolCall {
                        id: raw.id,
                        name: raw.name,
                        arguments: json!({ "_raw": raw.raw_args }),
                        malformed_error: Some(feedback),
                    });
                } else {
                    return Err(format!(
                        "exceeded malformed JSON recovery budget for tool `{}`: {parse_err}",
                        raw.name
                    ));
                }
            }
        }
    }

    if had_valid_call {
        recovery.reset();
    }
    Ok(prepared)
}

fn build_tool_call_assistant_message(
    thinking: String,
    thinking_signature: Option<String>,
    family: orangeagent_ai::ProviderFamily,
    text: String,
    calls: &[PreparedToolCall],
) -> Message {
    let mut content = Vec::new();
    if !thinking.is_empty() || thinking_signature.is_some() {
        content.push(ContentBlock::Thinking {
            text: thinking,
            signature: thinking_signature,
            provider: Some(family),
        });
    }
    if !text.is_empty() {
        content.push(ContentBlock::Text { text });
    }
    for call in calls {
        content.push(ContentBlock::ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        });
    }
    Message {
        role: orangeagent_ai::Role::Assistant,
        content,
    }
}

/// Group contiguous [`ConcurrencyMode::ReadOnlyParallel`] calls into parallel batches
/// while isolating [`ConcurrencyMode::ExclusiveSequential`] (and unknown/malformed) calls.
fn partition_tool_batches(calls: Vec<PreparedToolCall>, registry: &ToolRegistry) -> Vec<ToolBatch> {
    let mut batches = Vec::new();
    let mut current_parallel = Vec::new();

    for call in calls {
        let is_parallel = call.malformed_error.is_none()
            && registry
                .get(&call.name)
                .is_some_and(|t| t.concurrency_mode() == ConcurrencyMode::ReadOnlyParallel);

        if is_parallel {
            current_parallel.push(call);
        } else {
            if !current_parallel.is_empty() {
                batches.push(ToolBatch::Parallel(std::mem::take(&mut current_parallel)));
            }
            batches.push(ToolBatch::Sequential(call));
        }
    }

    if !current_parallel.is_empty() {
        batches.push(ToolBatch::Parallel(current_parallel));
    }
    batches
}

fn handle_turn_level_decision(
    decision: HookDecision,
    iterations: usize,
    total_usage: TokenUsage,
) -> Option<TurnOutcome> {
    match decision {
        HookDecision::PassThrough | HookDecision::Modify(_) => None,
        HookDecision::Deny(reason) => Some(TurnOutcome::Failed {
            error: format!("blocked by hook: {reason}"),
            iterations,
            total_usage,
        }),
        HookDecision::ShortCircuit(reason) => Some(TurnOutcome::Interrupted {
            reason,
            iterations,
            total_usage,
        }),
    }
}

fn emit(events: Option<&EventSink>, event: AgentEvent) {
    if let Some(tx) = events {
        let _ = tx.send(event);
    }
}

fn truncate_preview(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let preview: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{preview}...")
    } else {
        preview
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use async_trait::async_trait;
    use orangeagent_ai::{ProviderFamily, VcrInteraction, VcrItem, VcrProvider};
    use tokio::sync::mpsc;

    use crate::AgentHook;

    use super::*;

    fn replay_provider(turns: Vec<Vec<StreamEvent>>) -> Arc<dyn LlmProvider> {
        let interactions = turns
            .into_iter()
            .enumerate()
            .map(|(idx, events)| VcrInteraction {
                request_hash: format!("turn-{idx}"),
                model: "deepseek-chat".into(),
                initial_error: None,
                items: events
                    .into_iter()
                    .map(|event| VcrItem::Event { event })
                    .collect(),
            })
            .collect();
        Arc::new(VcrProvider::from_interactions(
            ProviderFamily::OpenAiChat,
            interactions,
            false,
        ))
    }

    struct EchoReadTool {
        delay: Duration,
        in_flight: Arc<AtomicUsize>,
        max_concurrency: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Tool for EchoReadTool {
        fn name(&self) -> &str {
            "read"
        }

        fn description(&self) -> &str {
            "read a file"
        }

        fn parameters_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            })
        }

        fn concurrency_mode(&self) -> ConcurrencyMode {
            ConcurrencyMode::ReadOnlyParallel
        }

        async fn execute(&self, args: Value, _ctx: &ToolContext) -> ToolOutput {
            let active = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_concurrency.fetch_max(active, Ordering::SeqCst);
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            let path = args
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("<missing>");
            ToolOutput::ok(format!("contents of {path}"))
        }
    }

    struct WriteTool {
        log: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl Tool for WriteTool {
        fn name(&self) -> &str {
            "write"
        }

        fn description(&self) -> &str {
            "write a file"
        }

        fn parameters_schema(&self) -> Value {
            json!({ "type": "object" })
        }

        async fn execute(&self, args: Value, _ctx: &ToolContext) -> ToolOutput {
            let label = args
                .get("label")
                .and_then(Value::as_str)
                .unwrap_or("w")
                .to_string();
            self.log.lock().expect("lock").push(label.clone());
            ToolOutput::ok(format!("wrote {label}"))
        }
    }

    #[tokio::test]
    async fn multi_turn_tool_loop_completes_with_vcr_replay() {
        let provider = replay_provider(vec![
            vec![
                StreamEvent::ToolCallStart {
                    index: 0,
                    id: "call_1".into(),
                    name: "read".into(),
                },
                StreamEvent::ToolCallArgsDelta {
                    index: 0,
                    json_fragment: r#"{"path":"Cargo.toml"}"#.into(),
                },
                StreamEvent::Usage(TokenUsage {
                    input_tokens: 100,
                    output_tokens: 20,
                    ..TokenUsage::default()
                }),
                StreamEvent::Done {
                    reason: FinishReason::ToolCalls,
                },
            ],
            vec![
                StreamEvent::TextDelta {
                    text: "Read Cargo.toml successfully.".into(),
                },
                StreamEvent::Usage(TokenUsage {
                    input_tokens: 150,
                    output_tokens: 10,
                    cache_read_tokens: 100,
                    ..TokenUsage::default()
                }),
                StreamEvent::Done {
                    reason: FinishReason::Stop,
                },
            ],
        ]);

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoReadTool {
            delay: Duration::ZERO,
            in_flight: Arc::new(AtomicUsize::new(0)),
            max_concurrency: Arc::new(AtomicUsize::new(0)),
        }));

        let agent = AgentLoop::new(
            provider,
            registry,
            CompositeHook::new(),
            LoopConfig::new("deepseek-chat"),
        );

        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut messages = vec![Message::user("Inspect Cargo.toml")];
        let outcome = agent
            .run_turn(
                "turn_1",
                &mut messages,
                &SteerQueue::new(),
                Some(&tx),
                CancellationToken::new(),
            )
            .await;

        assert_eq!(
            outcome,
            TurnOutcome::Completed {
                final_text: "Read Cargo.toml successfully.".into(),
                iterations: 2,
                total_usage: TokenUsage {
                    input_tokens: 250,
                    output_tokens: 30,
                    cache_read_tokens: 100,
                    cache_write_tokens: 0,
                    reasoning_tokens: 0,
                },
            }
        );
        // user + assistant(tool_call) + tool(result) + assistant(final)
        assert_eq!(messages.len(), 4);

        drop(tx);
        let mut collected = Vec::new();
        while let Some(ev) = rx.recv().await {
            collected.push(ev);
        }
        assert!(matches!(
            collected.first(),
            Some(AgentEvent::TurnStarted { .. })
        ));
        assert!(matches!(
            collected.last(),
            Some(AgentEvent::TurnFinished { .. })
        ));
    }

    #[tokio::test]
    async fn readonly_tools_run_in_parallel_and_writes_stay_sequential() {
        let provider = replay_provider(vec![
            vec![
                StreamEvent::ToolCallStart {
                    index: 0,
                    id: "c1".into(),
                    name: "read".into(),
                },
                StreamEvent::ToolCallArgsDelta {
                    index: 0,
                    json_fragment: r#"{"path":"a.rs"}"#.into(),
                },
                StreamEvent::ToolCallStart {
                    index: 1,
                    id: "c2".into(),
                    name: "read".into(),
                },
                StreamEvent::ToolCallArgsDelta {
                    index: 1,
                    json_fragment: r#"{"path":"b.rs"}"#.into(),
                },
                StreamEvent::ToolCallStart {
                    index: 2,
                    id: "c3".into(),
                    name: "read".into(),
                },
                StreamEvent::ToolCallArgsDelta {
                    index: 2,
                    json_fragment: r#"{"path":"c.rs"}"#.into(),
                },
                StreamEvent::ToolCallStart {
                    index: 3,
                    id: "c4".into(),
                    name: "write".into(),
                },
                StreamEvent::ToolCallArgsDelta {
                    index: 3,
                    json_fragment: r#"{"label":"first"}"#.into(),
                },
                StreamEvent::ToolCallStart {
                    index: 4,
                    id: "c5".into(),
                    name: "write".into(),
                },
                StreamEvent::ToolCallArgsDelta {
                    index: 4,
                    json_fragment: r#"{"label":"second"}"#.into(),
                },
                StreamEvent::Done {
                    reason: FinishReason::ToolCalls,
                },
            ],
            vec![
                StreamEvent::TextDelta {
                    text: "done".into(),
                },
                StreamEvent::Done {
                    reason: FinishReason::Stop,
                },
            ],
        ]);

        let max_concurrency = Arc::new(AtomicUsize::new(0));
        let write_log = Arc::new(Mutex::new(Vec::new()));

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoReadTool {
            delay: Duration::from_millis(50),
            in_flight: Arc::new(AtomicUsize::new(0)),
            max_concurrency: Arc::clone(&max_concurrency),
        }));
        registry.register(Arc::new(WriteTool {
            log: Arc::clone(&write_log),
        }));

        let agent = AgentLoop::new(
            provider,
            registry,
            CompositeHook::new(),
            LoopConfig::new("deepseek-chat"),
        );

        let started = Instant::now();
        let mut messages = vec![Message::user("batch ops")];
        let outcome = agent
            .run_turn(
                "turn_par",
                &mut messages,
                &SteerQueue::new(),
                None,
                CancellationToken::new(),
            )
            .await;
        let elapsed = started.elapsed();

        assert!(matches!(
            outcome,
            TurnOutcome::Completed { iterations: 2, .. }
        ));
        assert_eq!(max_concurrency.load(Ordering::SeqCst), 3);
        // 3 parallel 50ms reads should finish well under 140ms (sequential would be >= 150ms).
        assert!(elapsed < Duration::from_millis(140), "elapsed={elapsed:?}");
        assert_eq!(*write_log.lock().expect("lock"), vec!["first", "second"]);
    }

    #[tokio::test]
    async fn recovers_from_malformed_json_and_empty_response() {
        let provider = replay_provider(vec![
            // Iteration 1: empty response -> triggers empty recovery
            vec![StreamEvent::Done {
                reason: FinishReason::Stop,
            }],
            // Iteration 2: malformed JSON tool args -> feeds error ToolResult back
            vec![
                StreamEvent::ToolCallStart {
                    index: 0,
                    id: "bad_1".into(),
                    name: "read".into(),
                },
                StreamEvent::ToolCallArgsDelta {
                    index: 0,
                    json_fragment: r#"{"path": "#.into(),
                },
                StreamEvent::Done {
                    reason: FinishReason::ToolCalls,
                },
            ],
            // Iteration 3: model fixes arguments and succeeds
            vec![
                StreamEvent::TextDelta {
                    text: "Recovered!".into(),
                },
                StreamEvent::Done {
                    reason: FinishReason::Stop,
                },
            ],
        ]);

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoReadTool {
            delay: Duration::ZERO,
            in_flight: Arc::new(AtomicUsize::new(0)),
            max_concurrency: Arc::new(AtomicUsize::new(0)),
        }));

        let agent = AgentLoop::new(
            provider,
            registry,
            CompositeHook::new(),
            LoopConfig::new("deepseek-chat"),
        );

        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut messages = vec![Message::user("recover please")];
        let outcome = agent
            .run_turn(
                "turn_rec",
                &mut messages,
                &SteerQueue::new(),
                Some(&tx),
                CancellationToken::new(),
            )
            .await;

        assert!(matches!(
            outcome,
            TurnOutcome::Completed {
                ref final_text,
                iterations: 3,
                ..
            } if final_text == "Recovered!"
        ));

        drop(tx);
        let mut recovery_events = 0usize;
        while let Some(ev) = rx.recv().await {
            if matches!(ev, AgentEvent::RecoveryTriggered { .. }) {
                recovery_events += 1;
            }
        }
        assert_eq!(recovery_events, 2);
    }

    #[tokio::test]
    async fn hook_deny_and_steer_injection_work_end_to_end() {
        struct DenyWriteHook;

        #[async_trait]
        impl AgentHook for DenyWriteHook {
            fn name(&self) -> &str {
                "deny-write"
            }

            async fn on_event(&self, event: &HookEvent) -> HookDecision {
                if event.phase == HookPhase::BeforeToolUse
                    && event.tool_name.as_deref() == Some("write")
                {
                    HookDecision::Deny("read-only mode".into())
                } else {
                    HookDecision::PassThrough
                }
            }
        }

        let provider = replay_provider(vec![
            vec![
                StreamEvent::ToolCallStart {
                    index: 0,
                    id: "w1".into(),
                    name: "write".into(),
                },
                StreamEvent::ToolCallArgsDelta {
                    index: 0,
                    json_fragment: r#"{"label":"blocked"}"#.into(),
                },
                StreamEvent::Done {
                    reason: FinishReason::ToolCalls,
                },
            ],
            vec![
                StreamEvent::TextDelta {
                    text: "Acknowledged denial and steer.".into(),
                },
                StreamEvent::Done {
                    reason: FinishReason::Stop,
                },
            ],
        ]);

        let write_log = Arc::new(Mutex::new(Vec::new()));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(WriteTool {
            log: Arc::clone(&write_log),
        }));

        let mut hooks = CompositeHook::new();
        hooks.push(Arc::new(DenyWriteHook));

        let steer = SteerQueue::new();
        steer.push("Also check tests please.");

        let agent = AgentLoop::new(provider, registry, hooks, LoopConfig::new("deepseek-chat"));
        let mut messages = vec![Message::user("try writing")];
        let outcome = agent
            .run_turn(
                "turn_hook",
                &mut messages,
                &steer,
                None,
                CancellationToken::new(),
            )
            .await;

        assert!(matches!(
            outcome,
            TurnOutcome::Completed { iterations: 2, .. }
        ));
        assert!(write_log.lock().expect("lock").is_empty());
        assert!(steer.is_empty());
        // Verify tool denial feedback and steer message were both appended to history.
        assert!(messages.iter().any(|m| {
            m.content.iter().any(|b| {
                matches!(b, ContentBlock::ToolResult { is_error: true, content, .. } if content.contains("read-only mode"))
            })
        }));
        assert!(
            messages
                .iter()
                .any(|m| m.text_content() == "Also check tests please.")
        );
    }

    #[tokio::test]
    async fn cancellation_token_interrupts_turn_cleanly() {
        let provider = replay_provider(vec![vec![StreamEvent::TextDelta {
            text: "never reached".into(),
        }]]);

        let agent = AgentLoop::new(
            provider,
            ToolRegistry::new(),
            CompositeHook::new(),
            LoopConfig::new("deepseek-chat"),
        );

        let cancel = CancellationToken::new();
        cancel.cancel();

        let mut messages = vec![Message::user("cancel me")];
        let outcome = agent
            .run_turn(
                "turn_cancel",
                &mut messages,
                &SteerQueue::new(),
                None,
                cancel,
            )
            .await;

        assert!(matches!(
            outcome,
            TurnOutcome::Interrupted { iterations: 0, .. }
        ));
    }
}
