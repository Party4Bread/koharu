use std::{fmt, sync::Arc};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use specta::Type;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    Codex, CodexModel, Config, Control, Host, HostCompletion, ProviderRetryPolicy, ToolCall,
    codex::{Delta, Request, function_output, message, project_context, retained_function_output},
    payload::{
        MODEL_RESULT_BUDGET_BYTES, PayloadCompaction, PayloadStore, TRACE_PAYLOAD_BUDGET_BYTES,
        release_observed_image_data,
    },
    provider::{duration_millis, transient_provider_error},
    trace::Trace,
};

const MAX_HOST_CONTINUATIONS_WITHOUT_PROGRESS: usize = 3;

const INSTRUCTIONS: &str = r#"You are Koharu Agent, operating manga translation projects inside Koharu.
The complete current project state, or an explicit null project when none is open, is supplied in a koharu_project_context block on every user turn.
Page images are intentionally omitted from that context. Call view_page only when visual inspection is needed, and only for relevant pages.
Use the provided tools whenever the user asks to inspect or change the project. Never claim a change succeeded unless its tool result says it succeeded.
Create or open a project before using tools that require an open project. Import and export only through explicitly supplied absolute paths.
All project changes are revisioned and reversible. Do not ask for permission. Do not produce a plan or expose internal steps; continue using tools until the request is complete or cannot be completed.
Do not invent entity identifiers. Preserve artwork and existing authored content unless the user asks to change them."#;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize, Type)]
#[serde(transparent)]
pub struct RunId(Uuid);

impl RunId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Clone, Debug, Deserialize, Serialize, Type)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Started {
        run: RunId,
        trace: crate::TraceLocation,
    },
    TextDelta {
        run: RunId,
        delta: String,
    },
    ReasoningDelta {
        run: RunId,
        delta: String,
    },
    ToolStarted {
        run: RunId,
        call_id: String,
        name: String,
    },
    ToolFinished {
        run: RunId,
        call_id: String,
        name: String,
        changed: bool,
        output: String,
    },
    Completed {
        run: RunId,
        message: String,
    },
    Failed {
        run: RunId,
        message: String,
    },
    Cancelled {
        run: RunId,
    },
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct RunResult {
    pub run: RunId,
    pub message: String,
}

pub struct Agent<H> {
    codex: Codex,
    host: Arc<H>,
    config: koharu_config::Config<Config>,
    history: Mutex<Vec<Value>>,
    serial: Mutex<()>,
}

impl<H> Agent<H>
where
    H: Host,
{
    fn record_host_trace(&self, trace: &mut Trace, payloads: &PayloadStore) -> Result<()> {
        for record in self.host.take_trace_records() {
            let kind = format!("host-trace-{}", record.event);
            let (data, compacted) = if trace_event_requires_envelope(&record.event) {
                payloads.compact_value(&kind, record.data, TRACE_PAYLOAD_BUDGET_BYTES)?
            } else {
                payloads.bound_value(&kind, record.data, TRACE_PAYLOAD_BUDGET_BYTES)?
            };
            trace.record(&record.event, data)?;
            record_payload_compaction(trace, compacted)?;
        }
        Ok(())
    }

    pub fn new(codex: Codex, host: H) -> Result<Self> {
        Ok(Self {
            codex,
            host: Arc::new(host),
            config: Config::load()?,
            history: Mutex::new(Vec::new()),
            serial: Mutex::new(()),
        })
    }

    pub fn codex(&self) -> &Codex {
        &self.codex
    }

    pub fn config(&self) -> Result<Config> {
        Ok(self.config.read()?.clone())
    }

    pub async fn models(&self) -> Result<Vec<CodexModel>> {
        self.codex.models().await
    }

    pub fn save_config(&self, config: Config) -> Result<Config> {
        let mut current = self.config.write()?;
        *current = config;
        let saved = current.clone();
        current.save()?;
        Ok(saved)
    }

    pub async fn clear(&self) {
        self.history.lock().await.clear();
    }

    #[tracing::instrument(skip_all)]
    pub async fn run<F>(
        &self,
        run: RunId,
        prompt: String,
        control: Control,
        provider_retry: ProviderRetryPolicy,
        trace_location: crate::TraceLocation,
        mut publish: F,
    ) -> Result<RunResult>
    where
        F: FnMut(Event) + Send,
    {
        let _serial = self.serial.lock().await;
        if trace_location.run != run {
            anyhow::bail!("agent trace location belongs to a different run");
        }
        let payloads = PayloadStore::for_trace(trace_location.path())?;
        let mut trace = Trace::create(&trace_location)?;
        publish(Event::Started {
            run,
            trace: trace_location.clone(),
        });
        let initial_context = self.host.context().await;
        let (trace_initial_context, initial_compaction) = match initial_context.as_ref() {
            Ok(context) => {
                let (context, compacted) = payloads.compact_value(
                    "initial-project-context",
                    context.clone(),
                    TRACE_PAYLOAD_BUDGET_BYTES,
                )?;
                (Some(context), compacted)
            }
            Err(_) => (None, None),
        };
        trace.record(
            "run_started",
            json!({
                "prompt": &prompt,
                "project_state_before": trace_initial_context,
                "project_state_before_error": initial_context
                    .as_ref()
                    .err()
                    .map(|error| format!("{error:#}")),
                "image_capture_policy": "view_page image bytes are not persisted; each rendered attachment records its BLAKE3 content hash, media type, and byte length",
                "provider_retry": provider_retry.trace_value(),
            }),
        )?;
        record_payload_compaction(&mut trace, initial_compaction)?;
        let result = match initial_context {
            Ok(context) => {
                self.run_inner(
                    run,
                    prompt,
                    context,
                    &control,
                    provider_retry,
                    &mut trace,
                    &payloads,
                    &mut publish,
                )
                .await
            }
            Err(error) => Err(error.context("failed to capture initial project state")),
        };
        let final_context = self.host.context().await;
        let (trace_final_context, final_compaction) = match final_context.as_ref() {
            Ok(context) => {
                let (context, compacted) = payloads.compact_value(
                    "final-project-context",
                    context.clone(),
                    TRACE_PAYLOAD_BUDGET_BYTES,
                )?;
                (Some(context), compacted)
            }
            Err(_) => (None, None),
        };
        let terminal = match (&result, &final_context) {
            (Ok(result), Ok(_)) => trace.record(
                "run_completed",
                json!({ "message": &result.message, "project_state_after": trace_final_context }),
            ),
            (Err(error), Ok(context)) if control.is_cancelled() => trace.record(
                "run_cancelled",
                json!({ "error": format!("{error:#}"), "project_state_after": trace_final_context }),
            ),
            (Err(error), Ok(_)) => trace.record(
                "run_failed",
                json!({ "error": format!("{error:#}"), "project_state_after": trace_final_context }),
            ),
            (Ok(_), Err(error)) | (Err(_), Err(error)) => trace.record(
                "run_failed",
                json!({
                    "error": format!("failed to capture final project state: {error:#}"),
                    "project_state_after": null,
                }),
            ),
        };
        record_payload_compaction(&mut trace, final_compaction)?;
        let result = match (result, final_context, terminal) {
            (_, _, Err(error)) => Err(error.context("failed to persist the agent trace")),
            (Ok(result), Ok(_), Ok(())) => Ok(result),
            (Err(error), Ok(_), Ok(())) => Err(error),
            (_, Err(error), Ok(())) => Err(error.context("failed to capture final project state")),
        };
        match &result {
            Ok(result) => publish(Event::Completed {
                run,
                message: result.message.clone(),
            }),
            Err(_) if control.is_cancelled() => publish(Event::Cancelled { run }),
            Err(error) => publish(Event::Failed {
                run,
                message: format!("{error:#}"),
            }),
        }
        result
    }

    async fn run_inner<F>(
        &self,
        run: RunId,
        prompt: String,
        context: Value,
        control: &Control,
        provider_retry: ProviderRetryPolicy,
        trace: &mut Trace,
        payloads: &PayloadStore,
        publish: &mut F,
    ) -> Result<RunResult>
    where
        F: FnMut(Event),
    {
        control.ensure_running()?;
        let config = self.config()?;
        let models = self.codex.models().await?;
        let model = match config.model.as_deref() {
            Some(selected) => models
                .iter()
                .find(|model| model.id == selected)
                .with_context(|| format!("configured Codex model {selected} is not available"))?,
            None => models
                .first()
                .context("Codex returned no available model")?,
        };
        let reasoning = if model.reasoning.is_empty() || model.reasoning.contains(&config.reasoning)
        {
            config.reasoning
        } else {
            model.reasoning[0]
        };
        let clean_user = message("user", &prompt);
        let (context, context_compaction) =
            payloads.compact_value("model-project-context", context, MODEL_RESULT_BUDGET_BYTES)?;
        record_payload_compaction(trace, context_compaction)?;
        let context_message = project_context(&context)?;
        let base = self.history.lock().await.clone();
        let mut input = base.clone();
        input.push(context_message);
        input.push(clean_user.clone());
        let mut persisted = base;
        persisted.push(clean_user);
        let session = run.to_string();
        let mut host_continuations_without_progress = 0;
        let mut last_host_progress_marker = None;

        loop {
            control.ensure_running()?;
            let transcript_compaction = payloads.bound_transcript(&mut input)?;
            record_payload_compaction(trace, transcript_compaction)?;
            // Tool availability is host workflow state, not run-global configuration. Hosts may
            // deliberately narrow or advance the surface after each invocation.
            let tools = self.host.tools();
            self.record_host_trace(trace, payloads)?;
            let request = Request::new(
                model.id.clone(),
                INSTRUCTIONS.to_owned(),
                input.clone(),
                tools.clone(),
                reasoning,
                session.clone(),
            );
            let mut attempt = 1;
            let turn = loop {
                let mut trace_error = None;
                let response = self
                    .codex
                    .respond(&request, control, |event| match event {
                        Delta::Text(delta) => {
                            if trace_error.is_none()
                                && let Err(error) =
                                    trace.record("model_text_delta", json!({ "delta": &delta }))
                            {
                                trace_error = Some(error);
                            }
                            publish(Event::TextDelta { run, delta });
                        }
                        Delta::Reasoning(delta) => {
                            if trace_error.is_none()
                                && let Err(error) = trace
                                    .record("model_reasoning_delta", json!({ "delta": &delta }))
                            {
                                trace_error = Some(error);
                            }
                            publish(Event::ReasoningDelta { run, delta });
                        }
                    })
                    .await;
                if let Some(error) = trace_error {
                    return Err(error.context("failed to persist a model delta"));
                }
                match response {
                    Ok(turn) => break turn,
                    Err(error) => {
                        control.ensure_running()?;
                        let Some(class) = transient_provider_error(&error) else {
                            return Err(error);
                        };
                        if attempt >= provider_retry.max_attempts() {
                            trace.record(
                                "provider_retry_exhausted",
                                json!({
                                    "schema_version": 1,
                                    "error_class": class,
                                    "attempt": attempt,
                                    "max_attempts": provider_retry.max_attempts(),
                                    "delay_ms": null,
                                }),
                            )?;
                            anyhow::bail!(
                                "provider retry budget exhausted after {attempt} attempts (last error class: {})",
                                class.as_str(),
                            );
                        }
                        let delay = provider_retry.delay_after(attempt);
                        trace.record(
                            "provider_retry_scheduled",
                            json!({
                                "schema_version": 1,
                                "error_class": class,
                                "attempt": attempt,
                                "next_attempt": attempt + 1,
                                "max_attempts": provider_retry.max_attempts(),
                                "delay_ms": duration_millis(delay),
                            }),
                        )?;
                        tokio::select! {
                            () = tokio::time::sleep(delay) => {}
                            () = control.cancelled() => control.ensure_running()?,
                        }
                        control.ensure_running()?;
                        attempt += 1;
                    }
                }
            };
            if let Some((original_bytes, final_bytes)) = release_observed_image_data(&mut input)? {
                trace.record(
                    "payload_compacted",
                    json!({
                        "schema_version": 1,
                        "payload_kind": "retained_image_data",
                        "original_bytes": original_bytes,
                        "final_bytes": final_bytes,
                        "artifact": null,
                        "reason": "image bytes were released after model observation; tool artifact path and BLAKE3 provenance remain retained",
                    }),
                )?;
            }
            input.extend(turn.output.iter().cloned());
            persisted.extend(turn.output.iter().cloned());

            if turn.calls.is_empty() {
                let completion = self.host.completion().await;
                self.record_host_trace(trace, payloads)?;
                match completion? {
                    HostCompletion::Completed => {
                        let history_compaction = payloads.bound_transcript(&mut persisted)?;
                        record_payload_compaction(trace, history_compaction)?;
                        *self.history.lock().await = persisted;
                        return Ok(RunResult {
                            run,
                            message: turn.text,
                        });
                    }
                    HostCompletion::Continue {
                        phase,
                        exposed_tools,
                        reason,
                        progress_marker,
                    } => {
                        let progress_marker_changed = last_host_progress_marker
                            .as_ref()
                            .is_some_and(|last| last != &progress_marker);
                        if progress_marker_changed {
                            host_continuations_without_progress = 0;
                        } else {
                            host_continuations_without_progress += 1;
                        }
                        last_host_progress_marker = Some(progress_marker);
                        if host_continuations_without_progress
                            > MAX_HOST_CONTINUATIONS_WITHOUT_PROGRESS
                        {
                            trace.record(
                                "host_continuation_exhausted",
                                json!({
                                    "schema_version": 2,
                                    "consecutive_without_progress": host_continuations_without_progress,
                                    "max_consecutive_without_progress": MAX_HOST_CONTINUATIONS_WITHOUT_PROGRESS,
                                    "progress_marker_changed": progress_marker_changed,
                                    "phase": &phase,
                                    "exposed_tools": &exposed_tools,
                                    "reason": &reason,
                                    "last_model_message": &turn.text,
                                }),
                            )?;
                            anyhow::bail!(
                                "host continuation stopped after {MAX_HOST_CONTINUATIONS_WITHOUT_PROGRESS} consecutive requests without host-reported progress; phase: {phase}; exposed tools: {}; host reason: {reason}; last model message: {:?}",
                                exposed_tools.join(", "),
                                turn.text,
                            );
                        }
                        trace.record(
                            "host_continuation_requested",
                            json!({
                                "schema_version": 2,
                                "consecutive_without_progress": host_continuations_without_progress,
                                "max_consecutive_without_progress": MAX_HOST_CONTINUATIONS_WITHOUT_PROGRESS,
                                "progress_marker_changed": progress_marker_changed,
                                "phase": &phase,
                                "exposed_tools": &exposed_tools,
                                "reason": &reason,
                                "last_model_message": &turn.text,
                            }),
                        )?;
                        let exposed_tools = if exposed_tools.is_empty() {
                            "(none)".to_owned()
                        } else {
                            exposed_tools.join(", ")
                        };
                        let continuation = message(
                            "user",
                            format!(
                                "Host continuation required. Phase: {phase}. Exposed tools: {exposed_tools}. Reason: {reason} Continue using the exposed tools."
                            ),
                        );
                        input.push(continuation.clone());
                        persisted.push(continuation);
                        continue;
                    }
                }
            }

            for call in turn.calls {
                control.ensure_running()?;
                let arguments = serde_json::from_str::<Value>(&call.arguments)
                    .unwrap_or_else(|_| Value::String(call.arguments.clone()));
                let (tool_started, compacted) = payloads.bound_value(
                    &format!("tool-arguments-{}", call.name),
                    json!({
                        "call_id": &call.call_id,
                        "name": &call.name,
                        "arguments": arguments,
                        "arguments_json": &call.arguments,
                    }),
                    TRACE_PAYLOAD_BUDGET_BYTES,
                )?;
                trace.record("tool_started", tool_started)?;
                record_payload_compaction(trace, compacted)?;
                publish(Event::ToolStarted {
                    run,
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                });
                let invocation = self
                    .host
                    .invoke(
                        ToolCall {
                            call_id: call.call_id.clone(),
                            name: call.name.clone(),
                            arguments: call.arguments,
                        },
                        control,
                    )
                    .await;
                self.record_host_trace(trace, payloads)?;
                let (output, changed, images) = match invocation {
                    Ok(invocation) => {
                        let result_kind = format!("tool-result-{}", call.name);
                        let (value, compacted) = if tool_requires_envelope(&call.name) {
                            payloads.compact_value(
                                &result_kind,
                                invocation.value,
                                MODEL_RESULT_BUDGET_BYTES,
                            )?
                        } else {
                            payloads.bound_value(
                                &result_kind,
                                invocation.value,
                                MODEL_RESULT_BUDGET_BYTES,
                            )?
                        };
                        trace.record(
                            "tool_finished",
                            json!({
                                "call_id": &call.call_id,
                                "name": &call.name,
                                "result": &value,
                                "error": null,
                                "changed": invocation.changed,
                                "image_provenance": invocation
                                    .images
                                    .iter()
                                    .map(|image| json!({
                                        "label": &image.label,
                                        "provenance": &image.provenance,
                                    }))
                                    .collect::<Vec<_>>(),
                            }),
                        )?;
                        record_payload_compaction(trace, compacted)?;
                        (
                            json!({ "ok": true, "value": value }),
                            invocation.changed,
                            invocation.images,
                        )
                    }
                    Err(error) => {
                        let error = format!("{error:#}");
                        trace.record(
                            "tool_finished",
                            json!({
                                "call_id": &call.call_id,
                                "name": &call.name,
                                "result": null,
                                "error": &error,
                                "changed": false,
                                "image_provenance": [],
                            }),
                        )?;
                        (json!({ "ok": false, "error": error }), false, Vec::new())
                    }
                };
                publish(Event::ToolFinished {
                    run,
                    call_id: call.call_id.clone(),
                    name: call.name,
                    changed,
                    output: output.to_string(),
                });
                input.push(function_output(&call.call_id, &output, &images)?);
                persisted.push(retained_function_output(&call.call_id, &output, &images)?);
            }
        }
    }
}

fn record_payload_compaction(
    trace: &mut Trace,
    compacted: Option<PayloadCompaction>,
) -> Result<()> {
    if let Some(compacted) = compacted {
        trace.record(
            "payload_compacted",
            json!({
                "schema_version": 1,
                "payload_kind": compacted.kind,
                "original_bytes": compacted.original_bytes,
                "final_bytes": compacted.final_bytes,
                "artifact": compacted.artifact,
            }),
        )?;
    }
    Ok(())
}

fn tool_requires_envelope(name: &str) -> bool {
    matches!(
        name,
        "inspect_project"
            | "inspect_source_evidence"
            | "view_page_source_debug"
            | "classify_difficult_sfx"
            | "verify_ui_panel_anchor"
            | "inspect_page_evidence"
            | "run_source_analysis"
            | "run_pipeline"
            | "review_pages"
            | "submit_visual_semantic_review"
    )
}

fn trace_event_requires_envelope(event: &str) -> bool {
    matches!(
        event,
        "source_analysis"
            | "source_evidence_inspection"
            | "page_source_debug_view"
            | "page_translation_review"
            | "page_debug_view"
            | "free_dialogue_anchor_detection"
    )
}

#[cfg(test)]
mod tests {
    use std::{
        path::Path,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    use async_trait::async_trait;
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        HostCompletion, Invocation, Tool,
        codex::{ScriptedCodexHandle, ScriptedResponse, Turn},
        provider::ProviderErrorClass,
    };

    #[derive(Clone, Default)]
    struct CountingHost {
        invocations: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Host for CountingHost {
        async fn context(&self) -> Result<Value> {
            Ok(json!({ "revision": self.invocations.load(Ordering::SeqCst) }))
        }

        fn tools(&self) -> Vec<Tool> {
            vec![Tool::new(
                "mutate",
                "mutate once",
                json!({ "type": "object" }),
            )]
        }

        async fn invoke(&self, call: ToolCall, _control: &Control) -> Result<Invocation> {
            anyhow::ensure!(call.name == "mutate", "unexpected tool");
            self.invocations.fetch_add(1, Ordering::SeqCst);
            Invocation::changed(json!({ "revision": 1 }))
        }

        async fn completion(&self) -> Result<HostCompletion> {
            anyhow::ensure!(
                self.invocations.load(Ordering::SeqCst) == 1,
                "workflow did not mutate exactly once"
            );
            Ok(HostCompletion::Completed)
        }
    }

    #[derive(Clone, Default)]
    struct ActionableCompletionHost {
        completion_checks: Arc<AtomicUsize>,
        always_continue: bool,
    }

    #[async_trait]
    impl Host for ActionableCompletionHost {
        async fn context(&self) -> Result<Value> {
            Ok(json!({ "revision": 7 }))
        }

        fn tools(&self) -> Vec<Tool> {
            vec![Tool::new(
                "revise_page_translation",
                "revise rejected translation wording",
                json!({ "type": "object" }),
            )]
        }

        async fn invoke(&self, _call: ToolCall, _control: &Control) -> Result<Invocation> {
            unreachable!("this regression exercises model final text")
        }

        async fn completion(&self) -> Result<HostCompletion> {
            if self.always_continue || self.completion_checks.fetch_add(1, Ordering::SeqCst) == 0 {
                return Ok(HostCompletion::Continue {
                    phase: "semantic_revision".to_owned(),
                    exposed_tools: vec!["revise_page_translation".to_owned()],
                    reason: "page review rejected inaccurate wording".to_owned(),
                    progress_marker: "scene-revision:7".to_owned(),
                });
            }
            Ok(HostCompletion::Completed)
        }
    }

    #[derive(Clone)]
    struct SerialExactRepairHost {
        pending_blockers: Arc<AtomicUsize>,
        progress_marker: Arc<AtomicUsize>,
    }

    impl Default for SerialExactRepairHost {
        fn default() -> Self {
            Self {
                pending_blockers: Arc::new(AtomicUsize::new(5)),
                progress_marker: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl Host for SerialExactRepairHost {
        async fn context(&self) -> Result<Value> {
            Ok(json!({
                "pending_blockers": self.pending_blockers.load(Ordering::SeqCst),
                "progress_marker": self.progress_marker.load(Ordering::SeqCst),
            }))
        }

        fn tools(&self) -> Vec<Tool> {
            vec![Tool::new(
                "commit_exact_repair",
                "commit one exact deterministic repair",
                json!({ "type": "object" }),
            )]
        }

        async fn invoke(&self, call: ToolCall, _control: &Control) -> Result<Invocation> {
            anyhow::ensure!(call.name == "commit_exact_repair", "unexpected tool");
            let pending = self
                .pending_blockers
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |pending| {
                    pending.checked_sub(1)
                })
                .map_err(|_| anyhow::anyhow!("no pending blocker to repair"))?
                - 1;
            let progress_marker = self.progress_marker.fetch_add(1, Ordering::SeqCst) + 1;
            Invocation::changed(json!({
                "pending_blockers": pending,
                "progress_marker": progress_marker,
            }))
        }

        async fn completion(&self) -> Result<HostCompletion> {
            let pending = self.pending_blockers.load(Ordering::SeqCst);
            if pending > 0 {
                return Ok(HostCompletion::Continue {
                    phase: "deterministic_repair".to_owned(),
                    exposed_tools: vec!["commit_exact_repair".to_owned()],
                    reason: format!("{pending} exact repair blocker(s) remain"),
                    progress_marker: format!(
                        "scene-revision:{}",
                        self.progress_marker.load(Ordering::SeqCst)
                    ),
                });
            }
            anyhow::ensure!(
                self.progress_marker.load(Ordering::SeqCst) == 5,
                "workflow did not commit all five exact repairs"
            );
            Ok(HostCompletion::Completed)
        }
    }

    fn tool_turn() -> Turn {
        Turn {
            output: vec![json!({
                "type": "function_call",
                "call_id": "mutation-1",
                "name": "mutate",
                "arguments": "{}",
            })],
            calls: vec![ToolCall {
                call_id: "mutation-1".to_owned(),
                name: "mutate".to_owned(),
                arguments: "{}".to_owned(),
            }],
            text: String::new(),
        }
    }

    fn completed_turn() -> Turn {
        Turn {
            output: Vec::new(),
            calls: Vec::new(),
            text: "done".to_owned(),
        }
    }

    fn final_text_turn(text: &str) -> Turn {
        Turn {
            output: vec![message("assistant", text)],
            calls: Vec::new(),
            text: text.to_owned(),
        }
    }

    fn exact_repair_turn(sequence: usize) -> Turn {
        let call_id = format!("exact-repair-{sequence}");
        Turn {
            output: vec![json!({
                "type": "function_call",
                "call_id": &call_id,
                "name": "commit_exact_repair",
                "arguments": "{}",
            })],
            calls: vec![ToolCall {
                call_id,
                name: "commit_exact_repair".to_owned(),
                arguments: "{}".to_owned(),
            }],
            text: String::new(),
        }
    }

    fn test_agent(codex: Codex, host: CountingHost) -> Agent<CountingHost> {
        Agent {
            codex,
            host: Arc::new(host),
            config: koharu_config::Config::memory(Config::default()),
            history: Mutex::new(Vec::new()),
            serial: Mutex::new(()),
        }
    }

    fn test_trace(directory: &tempfile::TempDir, run: RunId) -> crate::TraceLocation {
        crate::TraceLocation::explicit(run, directory.path().join("trace.jsonl")).unwrap()
    }

    fn trace_records(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    async fn run_script(
        responses: Vec<ScriptedResponse>,
        host: CountingHost,
        policy: ProviderRetryPolicy,
        control: Control,
    ) -> (
        Result<RunResult>,
        ScriptedCodexHandle,
        tempfile::TempDir,
        crate::TraceLocation,
    ) {
        let (codex, handle) = Codex::scripted(responses);
        let agent = test_agent(codex, host);
        let directory = tempfile::tempdir().unwrap();
        let run = RunId::new();
        let trace = test_trace(&directory, run);
        let result = agent
            .run(
                run,
                "test".to_owned(),
                control,
                policy,
                trace.clone(),
                |_| {},
            )
            .await;
        (result, handle, directory, trace)
    }

    #[tokio::test]
    async fn actionable_host_completion_requests_another_model_turn() {
        let host = ActionableCompletionHost::default();
        let checks = Arc::clone(&host.completion_checks);
        let (codex, handle) = Codex::scripted([
            ScriptedResponse::Turn(final_text_turn("I could not finish.")),
            ScriptedResponse::Turn(final_text_turn("Finished after revision.")),
        ]);
        let agent = Agent {
            codex,
            host: Arc::new(host),
            config: koharu_config::Config::memory(Config::default()),
            history: Mutex::new(Vec::new()),
            serial: Mutex::new(()),
        };
        let directory = tempfile::tempdir().unwrap();
        let run = RunId::new();
        let trace = test_trace(&directory, run);

        let result = agent
            .run(
                run,
                "test".to_owned(),
                Control::default(),
                ProviderRetryPolicy::new(1, Duration::ZERO, Duration::ZERO).unwrap(),
                trace,
                |_| {},
            )
            .await
            .unwrap();

        assert_eq!(result.message, "Finished after revision.");
        assert_eq!(handle.calls(), 2);
        assert_eq!(checks.load(Ordering::SeqCst), 2);
        let requests = handle.requests();
        let continuation = requests[1]["input"].as_array().unwrap().last().unwrap();
        let continuation = continuation["content"][0]["text"].as_str().unwrap();
        assert!(continuation.contains("semantic_revision"));
        assert!(continuation.contains("revise_page_translation"));
        assert!(continuation.contains("page review rejected inaccurate wording"));
    }

    #[tokio::test]
    async fn five_serial_exact_repairs_complete_with_fresh_host_progress() {
        let host = SerialExactRepairHost::default();
        let pending_blockers = Arc::clone(&host.pending_blockers);
        let progress_marker = Arc::clone(&host.progress_marker);
        let responses = (1..=5)
            .flat_map(|sequence| {
                [
                    ScriptedResponse::Turn(exact_repair_turn(sequence)),
                    ScriptedResponse::Turn(final_text_turn(&format!(
                        "Committed exact repair {sequence}."
                    ))),
                ]
            })
            .collect::<Vec<_>>();
        let (codex, handle) = Codex::scripted(responses);
        let agent = Agent {
            codex,
            host: Arc::new(host),
            config: koharu_config::Config::memory(Config::default()),
            history: Mutex::new(Vec::new()),
            serial: Mutex::new(()),
        };
        let directory = tempfile::tempdir().unwrap();
        let run = RunId::new();
        let trace = test_trace(&directory, run);

        let result = agent
            .run(
                run,
                "repair every blocker".to_owned(),
                Control::default(),
                ProviderRetryPolicy::new(1, Duration::ZERO, Duration::ZERO).unwrap(),
                trace,
                |_| {},
            )
            .await
            .unwrap();

        assert_eq!(result.message, "Committed exact repair 5.");
        assert_eq!(handle.calls(), 10);
        assert_eq!(pending_blockers.load(Ordering::SeqCst), 0);
        assert_eq!(progress_marker.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn exhausted_host_continuations_report_the_actionable_state_honestly() {
        let host = ActionableCompletionHost {
            always_continue: true,
            ..Default::default()
        };
        let (codex, handle) = Codex::scripted([
            ScriptedResponse::Turn(final_text_turn("Stopped once.")),
            ScriptedResponse::Turn(final_text_turn("Stopped twice.")),
            ScriptedResponse::Turn(final_text_turn("Stopped three times.")),
            ScriptedResponse::Turn(final_text_turn("Still stopped.")),
        ]);
        let agent = Agent {
            codex,
            host: Arc::new(host),
            config: koharu_config::Config::memory(Config::default()),
            history: Mutex::new(Vec::new()),
            serial: Mutex::new(()),
        };
        let directory = tempfile::tempdir().unwrap();
        let run = RunId::new();
        let trace = test_trace(&directory, run);

        let error = agent
            .run(
                run,
                "test".to_owned(),
                Control::default(),
                ProviderRetryPolicy::new(1, Duration::ZERO, Duration::ZERO).unwrap(),
                trace.clone(),
                |_| {},
            )
            .await
            .unwrap_err();

        let error = format!("{error:#}");
        assert!(
            error.contains("stopped after 3 consecutive requests without host-reported progress")
        );
        assert!(error.contains("phase: semantic_revision"));
        assert!(error.contains("exposed tools: revise_page_translation"));
        assert!(error.contains("page review rejected inaccurate wording"));
        assert!(error.contains("last model message: \"Still stopped.\""));
        assert_eq!(handle.calls(), 4);
        let records = trace_records(Path::new(&trace.path));
        let exhausted = records
            .iter()
            .find(|record| record["event"] == "host_continuation_exhausted")
            .unwrap();
        assert_eq!(exhausted["data"]["phase"], "semantic_revision");
        assert_eq!(
            exhausted["data"]["exposed_tools"],
            json!(["revise_page_translation"])
        );
        assert_eq!(exhausted["data"]["last_model_message"], "Still stopped.");
    }

    #[tokio::test]
    async fn transient_failures_retry_in_place_without_replaying_a_tool_mutation() {
        let host = CountingHost::default();
        let invocations = Arc::clone(&host.invocations);
        let policy = ProviderRetryPolicy::new(3, Duration::ZERO, Duration::ZERO).unwrap();
        let (result, handle, _directory, trace) = run_script(
            vec![
                ScriptedResponse::Error(ProviderErrorClass::Overload),
                ScriptedResponse::Turn(tool_turn()),
                ScriptedResponse::Error(ProviderErrorClass::Timeout),
                ScriptedResponse::Turn(completed_turn()),
            ],
            host,
            policy,
            Control::default(),
        )
        .await;

        assert_eq!(result.unwrap().message, "done");
        assert_eq!(handle.calls(), 4);
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
        let requests = handle.requests();
        assert_eq!(requests[2], requests[3]);
        let records = trace_records(Path::new(&trace.path));
        assert_eq!(
            records
                .iter()
                .filter(|record| record["event"] == "provider_retry_scheduled")
                .count(),
            2
        );
        assert_eq!(
            records
                .iter()
                .filter(|record| record["event"] == "tool_started")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn invalid_auth_context_and_schema_errors_do_not_retry() {
        for class in [
            ProviderErrorClass::InvalidRequest,
            ProviderErrorClass::Authentication,
            ProviderErrorClass::ContextLimit,
            ProviderErrorClass::Schema,
        ] {
            let (result, handle, _directory, trace) = run_script(
                vec![ScriptedResponse::Error(class)],
                CountingHost::default(),
                ProviderRetryPolicy::new(4, Duration::ZERO, Duration::ZERO).unwrap(),
                Control::default(),
            )
            .await;

            assert!(result.is_err());
            assert_eq!(handle.calls(), 1, "unexpected retry for {class:?}");
            assert!(
                trace_records(Path::new(&trace.path))
                    .iter()
                    .all(|record| record["event"] != "provider_retry_scheduled")
            );
        }
    }

    #[tokio::test]
    async fn bounded_exhaustion_is_safe_in_the_error_and_trace() {
        let policy = ProviderRetryPolicy::new(3, Duration::ZERO, Duration::ZERO).unwrap();
        let (result, handle, _directory, trace) = run_script(
            vec![
                ScriptedResponse::Error(ProviderErrorClass::Overload),
                ScriptedResponse::Error(ProviderErrorClass::Overload),
                ScriptedResponse::Error(ProviderErrorClass::Overload),
            ],
            CountingHost::default(),
            policy,
            Control::default(),
        )
        .await;

        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("provider retry budget exhausted after 3 attempts"));
        assert!(!error.contains("servers are currently overloaded"));
        assert_eq!(handle.calls(), 3);
        let records = trace_records(Path::new(&trace.path));
        let exhausted = records
            .iter()
            .find(|record| record["event"] == "provider_retry_exhausted")
            .unwrap();
        assert_eq!(exhausted["data"]["error_class"], "overload");
        assert_eq!(exhausted["data"]["attempt"], 3);
        assert_eq!(exhausted["data"]["delay_ms"], Value::Null);
    }

    #[tokio::test]
    async fn cancellation_interrupts_retry_wait_without_another_request() {
        let (codex, handle) = Codex::scripted([
            ScriptedResponse::Error(ProviderErrorClass::Overload),
            ScriptedResponse::Turn(completed_turn()),
        ]);
        let agent = test_agent(codex, CountingHost::default());
        let directory = tempfile::tempdir().unwrap();
        let run = RunId::new();
        let trace = test_trace(&directory, run);
        let control = Control::default();
        let cancel = control.clone();
        let observed = handle.clone();
        let cancellation_trace = trace.path.clone();
        let canceller = tokio::spawn(async move {
            loop {
                let scheduled = std::fs::read_to_string(&cancellation_trace)
                    .map(|trace| trace.contains("provider_retry_scheduled"))
                    .unwrap_or(false);
                if observed.calls() == 1 && scheduled {
                    break;
                }
                tokio::task::yield_now().await;
            }
            cancel.cancel();
        });

        let result = agent
            .run(
                run,
                "test".to_owned(),
                control,
                ProviderRetryPolicy::new(2, Duration::from_secs(60), Duration::from_secs(60))
                    .unwrap(),
                trace.clone(),
                |_| {},
            )
            .await;
        canceller.await.unwrap();

        assert!(format!("{:#}", result.unwrap_err()).contains("agent run was cancelled"));
        assert_eq!(handle.calls(), 1);
        let records = trace_records(Path::new(&trace.path));
        assert!(
            records
                .iter()
                .any(|record| record["event"] == "run_cancelled")
        );
        assert!(
            records
                .iter()
                .all(|record| record["event"] != "provider_retry_exhausted")
        );
    }
}
