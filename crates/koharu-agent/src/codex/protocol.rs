use anyhow::{Context as _, Result, bail, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::Serialize;
use serde_json::{Value, json};

use crate::{Reasoning, Tool};

#[derive(Debug, Serialize)]
pub(crate) struct Request {
    pub(super) model: String,
    pub(super) instructions: String,
    pub(super) input: Vec<Value>,
    pub(super) tools: Vec<Tool>,
    pub(super) tool_choice: &'static str,
    pub(super) parallel_tool_calls: bool,
    pub(super) reasoning: ReasoningOptions,
    pub(super) text: TextOptions,
    pub(super) include: [&'static str; 1],
    pub(super) stream: bool,
    pub(super) store: bool,
    pub(super) prompt_cache_key: String,
}

#[derive(Debug, Serialize)]
pub(super) struct ReasoningOptions {
    effort: &'static str,
    summary: &'static str,
}

#[derive(Debug, Serialize)]
pub(super) struct TextOptions {
    verbosity: &'static str,
}

impl Request {
    pub(crate) fn new(
        model: String,
        instructions: String,
        input: Vec<Value>,
        tools: Vec<Tool>,
        reasoning: Reasoning,
        session: String,
    ) -> Self {
        Self {
            model,
            instructions,
            input,
            tools,
            tool_choice: "auto",
            parallel_tool_calls: false,
            reasoning: ReasoningOptions {
                effort: reasoning.as_str(),
                summary: "auto",
            },
            text: TextOptions { verbosity: "low" },
            include: ["reasoning.encrypted_content"],
            stream: true,
            store: false,
            prompt_cache_key: session,
        }
    }
}

pub(crate) fn message(role: &str, text: impl Into<String>) -> Value {
    json!({
        "type": "message",
        "role": role,
        "content": [{
            "type": "input_text",
            "text": text.into(),
        }],
    })
}

pub(crate) fn project_context(data: &Value) -> Result<Value, serde_json::Error> {
    let content = vec![json!({
        "type": "input_text",
        "text": format!(
            "<koharu_project_context>\n{}\n</koharu_project_context>",
            serde_json::to_string(data)?
        ),
    })];
    Ok(json!({
        "type": "message",
        "role": "user",
        "content": content,
    }))
}

pub(crate) fn function_output(
    call_id: &str,
    output: &Value,
    images: &[crate::ToolImage],
) -> Result<Value> {
    let output = if images.is_empty() {
        Value::String(serde_json::to_string(output)?)
    } else {
        let mut content = vec![json!({
            "type": "input_text",
            "text": serde_json::to_string(output)?,
        })];
        for image in images {
            validate_image_url(&image.data_url)?;
            content.push(json!({
                "type": "input_text",
                "text": retained_image_reference(image)?,
            }));
            content.push(json!({
                "type": "input_image",
                "image_url": image.data_url,
                "detail": "high",
            }));
        }
        Value::Array(content)
    };
    Ok(json!({
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }))
}

pub(crate) fn retained_function_output(
    call_id: &str,
    output: &Value,
    images: &[crate::ToolImage],
) -> Result<Value> {
    let output = if images.is_empty() {
        Value::String(serde_json::to_string(output)?)
    } else {
        Value::Array(
            std::iter::once(Ok(json!({
                "type": "input_text",
                "text": serde_json::to_string(output)?,
            })))
            .chain(images.iter().map(|image| {
                Ok(json!({
                    "type": "input_text",
                    "text": retained_image_reference(image)?,
                }))
            }))
            .collect::<Result<Vec<_>>>()?,
        )
    };
    Ok(json!({
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }))
}

fn retained_image_reference(image: &crate::ToolImage) -> Result<String> {
    Ok(format!(
        "<koharu_image_artifact>\n{}\n</koharu_image_artifact>",
        serde_json::to_string(&json!({
            "schema_version": 1,
            "label": image.label,
            "provenance": image.provenance,
            "image_bytes_retained": false,
        }))?
    ))
}

fn validate_image_url(value: &str) -> Result<()> {
    if let Some(data) = value.strip_prefix("data:") {
        let (metadata, payload) = data
            .split_once(',')
            .context("model image data URL is missing its payload separator")?;
        ensure!(
            metadata.starts_with("image/") && metadata.ends_with(";base64"),
            "model image data URL must contain a base64-encoded image media type"
        );
        ensure!(!payload.is_empty(), "model image data URL has no payload");
        STANDARD
            .decode(payload)
            .context("model image data URL contains invalid base64")?;
        return Ok(());
    }

    let url = reqwest::Url::parse(value).context("model image URL is invalid")?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        bail!("model image URL must use data:, http:, or https:");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{ToolImage, ToolImageProvenance};

    use super::*;

    fn image(url: &str) -> ToolImage {
        ToolImage {
            label: "debug render".to_owned(),
            data_url: url.to_owned(),
            provenance: ToolImageProvenance::ContentHash {
                algorithm: "blake3",
                digest: "digest".to_owned(),
                media_type: "image/png".to_owned(),
                byte_length: 3,
            },
        }
    }

    #[test]
    fn immediate_images_require_provider_supported_urls() {
        let local = image("/tmp/debug.png");
        assert!(function_output("call", &json!({}), &[local]).is_err());

        let data = image("data:image/png;base64,YWJj");
        let output = function_output("call", &json!({}), &[data]).unwrap();
        assert_eq!(
            output["output"][2]["image_url"],
            "data:image/png;base64,YWJj"
        );
    }

    #[test]
    fn retained_image_output_is_text_only() {
        let output = retained_function_output(
            "call",
            &json!({ "artifact": { "path": "/tmp/debug.png" } }),
            &[image("data:image/png;base64,YWJj")],
        )
        .unwrap();
        let serialized = serde_json::to_string(&output).unwrap();
        assert!(serialized.contains("koharu_image_artifact"));
        assert!(serialized.contains("/tmp/debug.png"));
        assert!(!serialized.contains("input_image"));
        assert!(!serialized.contains("image_url"));
    }
}
