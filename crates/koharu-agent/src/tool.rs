use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Control;

#[derive(Clone, Debug)]
pub struct HostTraceRecord {
    pub event: String,
    pub data: Value,
}

impl HostTraceRecord {
    #[must_use]
    pub fn new(event: impl Into<String>, data: Value) -> Self {
        Self {
            event: event.into(),
            data,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolImage {
    pub label: String,
    pub data_url: String,
    pub provenance: ToolImageProvenance,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolImageProvenance {
    ContentHash {
        algorithm: &'static str,
        digest: String,
        media_type: String,
        byte_length: usize,
    },
    Omitted {
        reason: String,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    kind: &'static str,
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub strict: bool,
}

impl Tool {
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            kind: "function",
            name: name.into(),
            description: description.into(),
            parameters,
            strict: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug)]
pub struct Invocation {
    pub value: Value,
    pub changed: bool,
    pub images: Vec<ToolImage>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HostCompletion {
    Completed,
    Continue {
        phase: String,
        exposed_tools: Vec<String>,
        reason: String,
        /// Opaque host-owned workflow state used only to detect objective progress between
        /// consecutive continuation requests.
        progress_marker: String,
    },
}

impl Invocation {
    pub fn read(value: impl Serialize) -> Result<Self> {
        Ok(Self {
            value: serde_json::to_value(value)?,
            changed: false,
            images: Vec::new(),
        })
    }

    pub fn changed(value: impl Serialize) -> Result<Self> {
        Ok(Self {
            value: serde_json::to_value(value)?,
            changed: true,
            images: Vec::new(),
        })
    }

    #[must_use]
    pub fn with_image(
        mut self,
        label: impl Into<String>,
        data_url: impl Into<String>,
        provenance: ToolImageProvenance,
    ) -> Self {
        self.images.push(ToolImage {
            label: label.into(),
            data_url: data_url.into(),
            provenance,
        });
        self
    }
}

#[async_trait]
pub trait Host: Send + Sync + 'static {
    async fn context(&self) -> Result<Value>;

    fn tools(&self) -> Vec<Tool>;

    async fn invoke(&self, call: ToolCall, control: &Control) -> Result<Invocation>;

    /// Return and remove structured records produced while handling the most recent host action.
    fn take_trace_records(&self) -> Vec<HostTraceRecord> {
        Vec::new()
    }

    /// Decide whether a model final message satisfies the host-owned workflow contract.
    /// An error is a terminal host failure; only `Continue` requests another model turn.
    async fn completion(&self) -> Result<HostCompletion> {
        Ok(HostCompletion::Completed)
    }
}
