use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, ensure};
use serde_json::{Map, Value, json};

use crate::codex::message;

pub const MODEL_RESULT_BUDGET_BYTES: usize = 48 * 1024;
pub const TRACE_PAYLOAD_BUDGET_BYTES: usize = 64 * 1024;
pub const RETAINED_TRANSCRIPT_BUDGET_BYTES: usize = 384 * 1024;
const MAX_INDEX_STRING_BYTES: usize = 512;
const MAX_CATALOG_VALUES: usize = 128;
const RETAINED_SUFFIX_ITEMS: usize = 6;

#[derive(Clone, Debug)]
pub struct PayloadCompaction {
    pub kind: String,
    pub original_bytes: usize,
    pub final_bytes: usize,
    pub artifact: Value,
}

pub(crate) struct PayloadStore {
    root: PathBuf,
}

pub fn compact_model_payload(
    artifact_root: &Path,
    kind: &str,
    value: Value,
) -> Result<(Value, PayloadCompaction)> {
    let store = PayloadStore {
        root: artifact_root.to_owned(),
    };
    let (value, compacted) = store.compact_value(kind, value, MODEL_RESULT_BUDGET_BYTES)?;
    Ok((
        value,
        compacted.context("forced model payload compaction did not produce a record")?,
    ))
}

pub fn compact_retained_transcript(
    artifact_root: &Path,
    transcript: &mut Vec<Value>,
) -> Result<Option<PayloadCompaction>> {
    PayloadStore {
        root: artifact_root.to_owned(),
    }
    .bound_transcript(transcript)
}

pub fn retained_transcript_bytes(transcript: &[Value]) -> Result<usize> {
    retained_serialized_len(transcript)
}

impl PayloadStore {
    pub(crate) fn for_trace(trace_path: &Path) -> Result<Self> {
        let file_name = trace_path
            .file_name()
            .context("agent trace path has no file name")?
            .to_string_lossy();
        let root = trace_path
            .parent()
            .context("agent trace path has no parent")?
            .join(format!("{file_name}.artifacts"));
        Ok(Self { root })
    }

    pub(crate) fn bound_value(
        &self,
        kind: &str,
        value: Value,
        budget: usize,
    ) -> Result<(Value, Option<PayloadCompaction>)> {
        let bytes = serde_json::to_vec(&value)?;
        if bytes.len() <= budget {
            return Ok((value, None));
        }
        self.compact_value(kind, value, budget)
    }

    pub(crate) fn compact_value(
        &self,
        kind: &str,
        value: Value,
        budget: usize,
    ) -> Result<(Value, Option<PayloadCompaction>)> {
        let bytes = serde_json::to_vec(&value)?;
        let artifact = self.write_json(kind, &bytes)?;
        let index = semantic_index(&value);
        let mut envelope = json!({
            "schema_version": 1,
            "payload_compacted": true,
            "payload_kind": kind,
            "artifact": artifact,
            "index": index,
        });
        if serialized_len(&envelope)? > budget {
            envelope["index"] = compact_index_catalog(&value);
        }
        let final_bytes = serialized_len(&envelope)?;
        ensure!(
            final_bytes <= budget,
            "compacted {kind} envelope is {final_bytes} bytes, above budget {budget}"
        );
        Ok((
            envelope,
            Some(PayloadCompaction {
                kind: kind.to_owned(),
                original_bytes: bytes.len(),
                final_bytes,
                artifact,
            }),
        ))
    }

    pub(crate) fn bound_transcript(
        &self,
        transcript: &mut Vec<Value>,
    ) -> Result<Option<PayloadCompaction>> {
        let original_bytes = retained_serialized_len(transcript)?;
        if original_bytes <= RETAINED_TRANSCRIPT_BUDGET_BYTES {
            return Ok(None);
        }

        let complete = retained_projection(&Value::Array(transcript.clone()));
        let complete_bytes = serde_json::to_vec(&complete)?;
        let artifact = self.write_json("retained-transcript", &complete_bytes)?;
        let mut suffix_start = transcript.len().saturating_sub(RETAINED_SUFFIX_ITEMS);
        if suffix_start > 0
            && transcript[suffix_start].get("type").and_then(Value::as_str)
                == Some("function_call_output")
        {
            suffix_start -= 1;
        }
        let prefix = retained_projection(&Value::Array(transcript[..suffix_start].to_vec()));
        let retained = json!({
            "schema_version": 1,
            "payload_compacted": true,
            "payload_kind": "retained_transcript",
            "artifact": artifact,
            "index": semantic_index(&prefix),
        });
        let mut compacted = vec![message(
            "user",
            format!(
                "<koharu_retained_transcript>\n{}\n</koharu_retained_transcript>",
                serde_json::to_string(&retained)?
            ),
        )];
        compacted.extend_from_slice(&transcript[suffix_start..]);
        if retained_serialized_len(&compacted)? > RETAINED_TRANSCRIPT_BUDGET_BYTES {
            let catalog = json!({
                "schema_version": 1,
                "payload_compacted": true,
                "payload_kind": "retained_transcript",
                "artifact": artifact,
                "index": compact_index_catalog(&complete),
            });
            compacted = vec![message(
                "user",
                format!(
                    "<koharu_retained_transcript>\n{}\n</koharu_retained_transcript>",
                    serde_json::to_string(&catalog)?
                ),
            )];
            compacted.extend(
                transcript
                    .iter()
                    .filter(|value| contains_input_image(value))
                    .cloned(),
            );
        }
        let final_bytes = retained_serialized_len(&compacted)?;
        ensure!(
            final_bytes <= RETAINED_TRANSCRIPT_BUDGET_BYTES,
            "compacted transcript is {final_bytes} bytes, above budget {}",
            RETAINED_TRANSCRIPT_BUDGET_BYTES
        );
        *transcript = compacted;
        Ok(Some(PayloadCompaction {
            kind: "retained_transcript".to_owned(),
            original_bytes,
            final_bytes,
            artifact,
        }))
    }

    fn write_json(&self, kind: &str, bytes: &[u8]) -> Result<Value> {
        std::fs::create_dir_all(&self.root)
            .with_context(|| format!("failed to create {}", self.root.display()))?;
        let digest = blake3::hash(bytes).to_hex().to_string();
        let mut stem = kind
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() {
                    character
                } else {
                    '-'
                }
            })
            .collect::<String>();
        stem.truncate(64);
        let path = self.root.join(format!("{stem}-{}.json", &digest[..16]));
        if !path.exists() {
            std::fs::write(&path, bytes)
                .with_context(|| format!("failed to write {}", path.display()))?;
        }
        Ok(json!({
            "path": path.to_string_lossy(),
            "media_type": "application/json",
            "byte_length": bytes.len(),
            "blake3": digest,
        }))
    }
}

pub(crate) fn retained_serialized_len(values: &[Value]) -> Result<usize> {
    Ok(serde_json::to_vec(&retained_projection(&Value::Array(values.to_vec())))?.len())
}

pub fn release_observed_image_data(values: &mut [Value]) -> Result<Option<(usize, usize)>> {
    let original = serialized_len(&Value::Array(values.to_vec()))?;
    let mut changed = false;
    for value in &mut *values {
        strip_images(value, &mut changed);
    }
    if !changed {
        return Ok(None);
    }
    let final_bytes = serialized_len(&Value::Array(values.to_vec()))?;
    Ok(Some((original, final_bytes)))
}

fn strip_images(value: &mut Value, changed: &mut bool) {
    match value {
        Value::Array(values) => {
            for value in values {
                strip_images(value, changed);
            }
        }
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("input_image") {
                let reference =
                    released_image_reference(object.get("image_url").and_then(Value::as_str));
                object.clear();
                object.insert("type".to_owned(), Value::String("input_text".to_owned()));
                object.insert("text".to_owned(), Value::String(reference));
                *changed = true;
                return;
            }
            for value in object.values_mut() {
                strip_images(value, changed);
            }
        }
        _ => {}
    }
}

fn released_image_reference(url: Option<&str>) -> String {
    let reference = match url {
        Some(url) if url.starts_with("data:") => url
            .split_once(',')
            .and_then(|(metadata, payload)| {
                use base64::{Engine as _, engine::general_purpose::STANDARD};
                STANDARD.decode(payload).ok().map(|bytes| {
                    json!({
                        "kind": "content_hash",
                        "algorithm": "blake3",
                        "digest": blake3::hash(&bytes).to_hex().to_string(),
                        "media_type": metadata
                            .strip_prefix("data:")
                            .and_then(|metadata| metadata.strip_suffix(";base64")),
                        "byte_length": bytes.len(),
                    })
                })
            })
            .unwrap_or_else(|| {
                json!({
                    "kind": "invalid_data_url_digest",
                    "algorithm": "blake3",
                    "digest": blake3::hash(url.as_bytes()).to_hex().to_string(),
                })
            }),
        Some(url) => json!({ "kind": "source_reference", "value": url }),
        None => json!({ "kind": "missing_source_reference" }),
    };
    format!(
        "<koharu_image_artifact>\n{}\n</koharu_image_artifact>",
        json!({
            "schema_version": 1,
            "provenance": reference,
            "image_bytes_retained": false,
        })
    )
}

fn contains_input_image(value: &Value) -> bool {
    match value {
        Value::Array(values) => values.iter().any(contains_input_image),
        Value::Object(object) => {
            object.get("type").and_then(Value::as_str) == Some("input_image")
                || object.values().any(contains_input_image)
        }
        _ => false,
    }
}

fn retained_projection(value: &Value) -> Value {
    let mut value = value.clone();
    let mut changed = false;
    strip_images(&mut value, &mut changed);
    value
}

fn semantic_index(value: &Value) -> Value {
    compact_node(value, None, false)
        .unwrap_or_else(|| json!({ "summary": "see complete artifact" }))
}

fn compact_node(value: &Value, key: Option<&str>, retain_primitives: bool) -> Option<Value> {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => {
            (retain_primitives || key.is_some_and(actionable_key)).then(|| value.clone())
        }
        Value::String(text) => (retain_primitives || key.is_some_and(actionable_key))
            .then(|| Value::String(short_text(text))),
        Value::Array(values) => {
            if key.is_some_and(|key| matches!(key, "candidates" | "rejected_candidates")) {
                return Some(candidate_summary(values));
            }
            let compacted = values
                .iter()
                .filter_map(|value| compact_node(value, key, retain_primitives))
                .collect::<Vec<_>>();
            (!compacted.is_empty()).then_some(Value::Array(compacted))
        }
        Value::Object(object) => {
            if key.is_some_and(|key| key.ends_with("bounds") || key.ends_with("bbox")) {
                let bounds = [
                    "x", "y", "width", "height", "left", "top", "right", "bottom",
                ]
                .into_iter()
                .filter_map(|field| {
                    object
                        .get(field)
                        .map(|value| (field.to_owned(), value.clone()))
                })
                .collect::<Map<_, _>>();
                return (!bounds.is_empty()).then_some(Value::Object(bounds));
            }
            if key.is_some_and(heavy_key) {
                return object
                    .get("bounds")
                    .and_then(|bounds| compact_node(bounds, Some("bounds"), true))
                    .map(|bounds| json!({ "bounds": bounds }));
            }
            let mut compacted = Map::new();
            for (child_key, child) in object {
                if heavy_key(child_key) {
                    if let Some(bounds) = child.get("bounds")
                        && let Some(bounds) = compact_node(bounds, Some("bounds"), true)
                    {
                        compacted.insert(format!("{child_key}_bounds"), bounds);
                    }
                    continue;
                }
                if let Some(child) = compact_node(
                    child,
                    Some(child_key),
                    retain_primitives || semantic_container(child_key),
                ) {
                    compacted.insert(child_key.clone(), child);
                }
            }
            (!compacted.is_empty()).then_some(Value::Object(compacted))
        }
    }
}

fn candidate_summary(values: &[Value]) -> Value {
    let mut accepted = Vec::new();
    let mut rejected_examples = Vec::new();
    let mut rejection_reason_counts = std::collections::BTreeMap::<String, usize>::new();
    for value in values {
        let compacted = compact_node(value, Some("candidate"), true);
        if value.get("accepted").and_then(Value::as_bool) == Some(true) {
            if let Some(compacted) = compacted {
                accepted.push(compacted);
            }
        } else if rejected_examples.len() < 3
            && let Some(compacted) = compacted
        {
            rejected_examples.push(compacted);
        }
        if let Some(reasons) = value.get("rejection_reasons").and_then(Value::as_array) {
            for reason in reasons.iter().filter_map(Value::as_str) {
                *rejection_reason_counts
                    .entry(short_text(reason))
                    .or_default() += 1;
            }
        }
    }
    let rejected_count = values.len().saturating_sub(accepted.len());
    json!({
        "candidate_count": values.len(),
        "accepted": accepted,
        "rejected_count": rejected_count,
        "rejection_reason_counts": rejection_reason_counts,
        "rejected_examples": rejected_examples,
    })
}

fn actionable_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    key == "id"
        || key.ends_with("_id")
        || key.contains("revision")
        || key.contains("ordinal")
        || key.contains("role")
        || key.contains("required")
        || key.contains("skip")
        || key.contains("group")
        || key.contains("anchor")
        || key.contains("decision")
        || key.contains("source")
        || key.contains("current")
        || key.contains("translation")
        || key.contains("language")
        || key.contains("bounds")
        || key.contains("path")
        || key.contains("media_type")
        || key.contains("byte_length")
        || key.contains("blake3")
        || key.contains("digest")
        || key.contains("candidate")
        || key.contains("reject")
        || key.contains("failure")
        || key.contains("issue")
        || key.contains("error")
        || key.contains("status")
        || key.contains("accepted")
        || key.contains("confidence")
        || key.contains("reason")
        || key.contains("next_action")
        || key.contains("count")
        || key.contains("contract")
        || key.contains("constraint")
        || key.contains("prompt")
        || matches!(key.as_str(), "text" | "label" | "kind" | "page")
}

fn semantic_container(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    key.contains("evidence")
        || key.contains("decision")
        || key.contains("judgment")
        || key.contains("repair_plan")
        || key.contains("next_action")
        || key.contains("rejection")
        || key.contains("issue")
        || key.contains("artifact")
        || key.contains("crop")
        || key.contains("membership")
        || key.contains("anchor")
        || key.contains("gate")
        || key.contains("measurement")
        || matches!(
            key.as_str(),
            "source_ocr"
                | "original_ocr"
                | "current_source"
                | "current_translation"
                | "target_translation"
        )
}

fn heavy_key(key: &str) -> bool {
    matches!(
        key,
        "points"
            | "pixels"
            | "pixel_analysis"
            | "raster"
            | "components"
            | "component_internals"
            | "alpha"
            | "mask"
            | "data"
            | "data_url"
            | "image_url"
            | "source_polygon"
            | "geometry"
            | "contour"
            | "border"
    )
}

fn short_text(text: &str) -> String {
    if text.len() <= MAX_INDEX_STRING_BYTES {
        return text.to_owned();
    }
    let mut end = MAX_INDEX_STRING_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

fn compact_index_catalog(value: &Value) -> Value {
    let mut revisions = BTreeSet::new();
    let mut pages = BTreeSet::new();
    let mut elements = BTreeSet::new();
    let mut ordinals = BTreeSet::new();
    let mut digests = BTreeSet::new();
    collect_catalog(
        value,
        None,
        &mut revisions,
        &mut pages,
        &mut elements,
        &mut ordinals,
        &mut digests,
    );
    let revisions_count = revisions.len();
    let pages_count = pages.len();
    let elements_count = elements.len();
    let ordinals_count = ordinals.len();
    let digests_count = digests.len();
    json!({
        "revisions": revisions.into_iter().take(MAX_CATALOG_VALUES).collect::<Vec<_>>(),
        "revision_count": revisions_count,
        "page_ids": pages.into_iter().take(MAX_CATALOG_VALUES).collect::<Vec<_>>(),
        "page_id_count": pages_count,
        "element_ids": elements.into_iter().take(MAX_CATALOG_VALUES).collect::<Vec<_>>(),
        "element_id_count": elements_count,
        "ordinals": ordinals.into_iter().take(MAX_CATALOG_VALUES).collect::<Vec<_>>(),
        "ordinal_count": ordinals_count,
        "digests": digests.into_iter().take(MAX_CATALOG_VALUES).collect::<Vec<_>>(),
        "digest_count": digests_count,
        "index_complete": revisions_count <= MAX_CATALOG_VALUES
            && pages_count <= MAX_CATALOG_VALUES
            && elements_count <= MAX_CATALOG_VALUES
            && ordinals_count <= MAX_CATALOG_VALUES
            && digests_count <= MAX_CATALOG_VALUES,
        "summary": "Complete payload and semantic bindings are retained in the referenced artifact; counts disclose any catalog truncation.",
    })
}

fn collect_catalog(
    value: &Value,
    key: Option<&str>,
    revisions: &mut BTreeSet<String>,
    pages: &mut BTreeSet<String>,
    elements: &mut BTreeSet<String>,
    ordinals: &mut BTreeSet<u64>,
    digests: &mut BTreeSet<String>,
) {
    match value {
        Value::Array(values) => {
            for value in values {
                collect_catalog(value, key, revisions, pages, elements, ordinals, digests);
            }
        }
        Value::Object(object) => {
            for (key, value) in object {
                collect_catalog(
                    value,
                    Some(key),
                    revisions,
                    pages,
                    elements,
                    ordinals,
                    digests,
                );
            }
        }
        Value::String(text) => match key.unwrap_or_default() {
            key if key.contains("revision") => {
                revisions.insert(short_text(text));
            }
            "page" | "page_id" => {
                pages.insert(short_text(text));
            }
            "element" | "element_id" => {
                elements.insert(short_text(text));
            }
            key if key.contains("blake3") || key.contains("digest") => {
                digests.insert(short_text(text));
            }
            _ => {}
        },
        Value::Number(number) => {
            if key.is_some_and(|key| key.contains("ordinal"))
                && let Some(value) = number.as_u64()
            {
                ordinals.insert(value);
            }
            if key.is_some_and(|key| key.contains("revision")) {
                revisions.insert(number.to_string());
            }
        }
        _ => {}
    }
}

fn serialized_len(value: &Value) -> Result<usize> {
    Ok(serde_json::to_vec(value)?.len())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn oversized_payload_keeps_ordinals_and_writes_complete_artifact() {
        let directory = tempdir().unwrap();
        let store = PayloadStore {
            root: directory.path().join("artifacts"),
        };
        let value = json!({
            "scene_revision": 9,
            "page_id": "page-1",
            "elements": (1..=20).map(|ordinal| json!({
                "ordinal": ordinal,
                "element_id": format!("element-{ordinal}"),
                "required": true,
                "current_source": { "text": "源", "language": "ja-JP" },
                "source_polygon": { "points": vec![json!({"x": 1, "y": 2}); 20_000], "bounds": {"x": 1, "y": 2, "width": 3, "height": 4} },
            })).collect::<Vec<_>>(),
        });
        let (bounded, event) = store
            .bound_value("test", value, MODEL_RESULT_BUDGET_BYTES)
            .unwrap();
        let event = event.unwrap();
        assert!(event.original_bytes > MODEL_RESULT_BUDGET_BYTES);
        assert!(event.final_bytes <= MODEL_RESULT_BUDGET_BYTES);
        assert_eq!(bounded["index"]["elements"][19]["ordinal"], 20);
        assert!(Path::new(bounded["artifact"]["path"].as_str().unwrap()).is_file());
    }
}
