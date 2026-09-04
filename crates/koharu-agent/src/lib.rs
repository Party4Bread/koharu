//! Koharu's in-process, OAuth-backed Codex agent.

mod agent;
mod codex;
mod config;
mod control;
mod payload;
mod provider;
mod tool;
mod trace;

pub use agent::{Agent, Event, Message, Role, RunId, RunResult};
pub use codex::{Account, Codex, CodexModel, LoginEvent};
pub use config::{Config, Reasoning};
pub use control::Control;
pub use payload::{
    MODEL_RESULT_BUDGET_BYTES, PayloadCompaction, RETAINED_TRANSCRIPT_BUDGET_BYTES,
    compact_model_payload, compact_retained_transcript, release_observed_image_data,
    retained_transcript_bytes,
};
pub use provider::{
    DEFAULT_PROVIDER_MAX_ATTEMPTS, DEFAULT_PROVIDER_RETRY_BASE_DELAY,
    DEFAULT_PROVIDER_RETRY_MAX_DELAY, ProviderRetryPolicy,
};
pub use tool::{
    Host, HostCompletion, HostTraceRecord, Invocation, Tool, ToolCall, ToolImage,
    ToolImageProvenance,
};
pub use trace::{TraceLocation, trace_location};
