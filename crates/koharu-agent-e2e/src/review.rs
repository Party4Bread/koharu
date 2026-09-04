use std::{fs::OpenOptions, io::Write as _, path::Path, process::Command};

use anyhow::{Context as _, Result, bail};
use koharu_scene::{EntityId, Revision};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::repair::{CorrectionRecord, DeterministicRepairPlan};

pub(crate) const VISUAL_REVIEW_SCHEMA_VERSION: u32 = 7;
pub(crate) const AGENT_VISUAL_SEMANTIC_REVIEW_SCHEMA_VERSION: u32 = 3;

pub(crate) struct ReviewPageInput {
    pub page_id: EntityId,
    pub label: String,
    pub original_media_type: String,
    pub original_bytes: Vec<u8>,
    pub preview_bytes: Vec<u8>,
    pub semantic_elements: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct VisualReviewRecord {
    pub schema_version: u32,
    pub attempt: u32,
    pub scene_revision: Revision,
    pub deterministic_acceptance_passed: bool,
    pub deterministic_repair_plan: DeterministicRepairPlan,
    pub status: VisualReviewStatus,
    pub bundle: ReviewBundle,
    pub judge: ExternalJudge,
    pub agent_reviews: Vec<AgentVisualSemanticReview>,
    pub decision: Option<VisualReviewDecision>,
    pub error: Option<String>,
}

impl VisualReviewRecord {
    pub(crate) fn accepted(&self) -> bool {
        if self.status != VisualReviewStatus::Accepted {
            return false;
        }
        if self.judge.required {
            return self
                .decision
                .as_ref()
                .is_some_and(VisualReviewDecision::passes);
        }
        // The host assembles this list from its page-indexed freshness state, so reviews from an
        // older scene revision are present only when that page's owned content has not changed.
        !self.bundle.pages.is_empty()
            && self.bundle.pages.iter().all(|page| {
                self.agent_reviews
                    .iter()
                    .any(|review| review.page_id == page.page_id && review.decision.passes())
            })
    }

    pub(crate) fn rejected(&self) -> bool {
        self.status == VisualReviewStatus::Rejected
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualReviewStatus {
    PendingAgentReview,
    Accepted,
    Rejected,
    Failed,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExternalJudge {
    pub required: bool,
    pub command: Option<String>,
    pub protocol: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct AgentVisualSemanticReview {
    pub schema_version: u32,
    pub page_id: EntityId,
    pub scene_revision: Revision,
    pub source_evidence_dossier_blake3: String,
    pub source_debug_artifact_blake3: String,
    pub dossier_blake3: String,
    pub debug_artifact_blake3: String,
    pub compacted_translation_reviews: Vec<CompactedTranslationSemanticReview>,
    pub decision: VisualReviewDecision,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompactedTranslationSemanticReview {
    pub logical_group: String,
    pub primary_render_element: String,
    pub member_evidence: Vec<CompactedTranslationMemberEvidence>,
    pub source_fidelity_preserved: bool,
    pub target_language_natural: bool,
    pub rationale: String,
    pub issues: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompactedTranslationMemberEvidence {
    pub ordinal: u32,
    pub element: String,
    pub source_crop_blake3: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReviewBundle {
    pub directory: String,
    pub manifest: String,
    pub pages: Vec<ReviewBundlePage>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReviewBundlePage {
    pub page_id: EntityId,
    pub label: String,
    pub original: ReviewArtifact,
    pub rendered_preview: ReviewArtifact,
    pub semantic_elements: ReviewArtifact,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReviewArtifact {
    pub path: String,
    pub media_type: String,
    pub byte_length: usize,
    pub blake3: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VisualReviewDecision {
    pub accepted: bool,
    pub summary: String,
    pub issues: Vec<String>,
    pub judgments: RequiredJudgments,
}

impl VisualReviewDecision {
    fn passes(&self) -> bool {
        self.accepted
            && self.judgments.all_pass()
            && self.issues.is_empty()
            && !self.summary.trim().is_empty()
    }

    pub(crate) fn validate_agent_submission(&self) -> Result<()> {
        if self.summary.trim().is_empty() {
            bail!("agent visual/semantic review summary cannot be empty");
        }
        if self.accepted {
            if !self.judgments.all_pass() {
                bail!(
                    "accepted agent visual/semantic review requires all seven judgments to be true"
                );
            }
            if !self.issues.is_empty() {
                bail!("accepted agent visual/semantic review cannot contain issues");
            }
        } else {
            if self.judgments.all_pass() {
                bail!(
                    "rejected agent visual/semantic review must mark at least one judgment false"
                );
            }
            if self.issues.iter().all(|issue| issue.trim().is_empty()) {
                bail!(
                    "rejected agent visual/semantic review must provide at least one concrete issue"
                );
            }
            if self.issues.iter().any(|issue| issue.trim().is_empty()) {
                bail!("agent visual/semantic review issues cannot contain empty entries");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredJudgments {
    pub source_text_accurate: bool,
    pub translation_meaning_accurate: bool,
    pub target_language_natural: bool,
    pub reading_order_preserved: bool,
    pub content_complete_without_duplicates: bool,
    pub typography_layout_acceptable: bool,
    pub skipped_items_are_difficult_sfx_and_no_required_content_skipped: bool,
}

impl RequiredJudgments {
    fn all_pass(&self) -> bool {
        self.source_text_accurate
            && self.translation_meaning_accurate
            && self.target_language_natural
            && self.reading_order_preserved
            && self.content_complete_without_duplicates
            && self.typography_layout_acceptable
            && self.skipped_items_are_difficult_sfx_and_no_required_content_skipped
    }
}

#[derive(Serialize)]
struct ReviewManifest<'a> {
    schema_version: u32,
    attempt: u32,
    scene_revision: Revision,
    purpose: &'static str,
    deterministic_acceptance_passed: bool,
    deterministic_repair_plan: &'a DeterministicRepairPlan,
    acceptance_policy: &'static str,
    line_break_review_policy: &'static str,
    required_judgments: [&'static str; 7],
    corrections: &'a [CorrectionRecord],
    pages: &'a [ReviewBundlePage],
}

pub(crate) fn run_visual_review(
    bundle_directory: &Path,
    judge_command: Option<&Path>,
    pages: Vec<ReviewBundlePage>,
    deterministic_acceptance_passed: bool,
    attempt: u32,
    scene_revision: Revision,
    corrections: Vec<CorrectionRecord>,
    deterministic_repair_plan: DeterministicRepairPlan,
) -> Result<VisualReviewRecord> {
    let bundle = write_bundle(
        bundle_directory,
        pages,
        deterministic_acceptance_passed,
        attempt,
        scene_revision,
        &deterministic_repair_plan,
        &corrections,
    )?;
    let judge = ExternalJudge {
        required: judge_command.is_some(),
        command: judge_command.map(path_string).transpose()?,
        protocol: "manifest path as argv[1]; one VisualReviewDecision JSON object on stdout with accepted, summary, issues, and all seven required boolean judgments",
    };
    let Some(command) = judge_command else {
        return Ok(VisualReviewRecord {
            schema_version: VISUAL_REVIEW_SCHEMA_VERSION,
            attempt,
            scene_revision,
            deterministic_acceptance_passed,
            deterministic_repair_plan,
            status: VisualReviewStatus::PendingAgentReview,
            bundle,
            judge,
            agent_reviews: Vec::new(),
            decision: None,
            error: None,
        });
    };
    let output = match Command::new(command).arg(&bundle.manifest).output() {
        Ok(output) => output,
        Err(error) => {
            return Ok(VisualReviewRecord {
                schema_version: VISUAL_REVIEW_SCHEMA_VERSION,
                attempt,
                scene_revision,
                deterministic_acceptance_passed,
                deterministic_repair_plan,
                status: VisualReviewStatus::Failed,
                bundle,
                judge,
                agent_reviews: Vec::new(),
                decision: None,
                error: Some(format!(
                    "failed to launch visual-review judge {}: {error}",
                    command.display()
                )),
            });
        }
    };
    if !output.status.success() {
        return Ok(VisualReviewRecord {
            schema_version: VISUAL_REVIEW_SCHEMA_VERSION,
            attempt,
            scene_revision,
            deterministic_acceptance_passed,
            deterministic_repair_plan,
            status: VisualReviewStatus::Failed,
            bundle,
            judge,
            agent_reviews: Vec::new(),
            decision: None,
            error: Some(format!(
                "visual-review judge exited with {}; stderr: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )),
        });
    }
    let decision = match serde_json::from_slice::<VisualReviewDecision>(&output.stdout) {
        Ok(decision) => decision,
        Err(error) => {
            return Ok(VisualReviewRecord {
                schema_version: VISUAL_REVIEW_SCHEMA_VERSION,
                attempt,
                scene_revision,
                deterministic_acceptance_passed,
                deterministic_repair_plan,
                status: VisualReviewStatus::Failed,
                bundle,
                judge,
                agent_reviews: Vec::new(),
                decision: None,
                error: Some(format!(
                    "visual-review judge stdout was not a valid decision: {error}"
                )),
            });
        }
    };
    let passes = decision.passes();
    let status = if passes {
        VisualReviewStatus::Accepted
    } else if !decision.summary.trim().is_empty() && !decision.issues.is_empty() {
        VisualReviewStatus::Rejected
    } else {
        return Ok(VisualReviewRecord {
            schema_version: VISUAL_REVIEW_SCHEMA_VERSION,
            attempt,
            scene_revision,
            deterministic_acceptance_passed,
            deterministic_repair_plan,
            status: VisualReviewStatus::Failed,
            bundle,
            judge,
            agent_reviews: Vec::new(),
            decision: Some(decision),
            error: Some(
                "command judge must provide a nonempty summary and concrete issues for every rejection"
                    .to_owned(),
            ),
        });
    };
    Ok(VisualReviewRecord {
        schema_version: VISUAL_REVIEW_SCHEMA_VERSION,
        attempt,
        scene_revision,
        deterministic_acceptance_passed,
        deterministic_repair_plan,
        status,
        bundle,
        judge,
        agent_reviews: Vec::new(),
        decision: Some(decision),
        error: None,
    })
}

fn write_bundle(
    bundle_directory: &Path,
    pages: Vec<ReviewBundlePage>,
    deterministic_acceptance_passed: bool,
    attempt: u32,
    scene_revision: Revision,
    deterministic_repair_plan: &DeterministicRepairPlan,
    corrections: &[CorrectionRecord],
) -> Result<ReviewBundle> {
    if bundle_directory.exists() {
        bail!(
            "visual-review bundle directory already exists: {}",
            bundle_directory.display()
        );
    }
    std::fs::create_dir_all(bundle_directory).with_context(|| {
        format!(
            "failed to create visual-review bundle {}",
            bundle_directory.display()
        )
    })?;
    let manifest_path = bundle_directory.join("manifest.json");
    let manifest = ReviewManifest {
        schema_version: VISUAL_REVIEW_SCHEMA_VERSION,
        attempt,
        scene_revision,
        purpose: "Visual and semantic fidelity review for either the constrained agent loop or a configured command judge. The deterministic acceptance result is included for context; rejected runs remain reviewable because geometry gates are not evidence of translation correctness or visual quality.",
        deterministic_acceptance_passed,
        deterministic_repair_plan,
        acceptance_policy: "Accept only when every explicit judgment passes and issues is empty. Deterministic layout evidence records font/glyph size, source/target-anchor occupancy and overflow, and contour clearance as diagnostics; these measurements are not independent rejection gates. Required text must still have nonempty source and target-language semantics, render successfully as finite positive visible glyphs, remain on the page, and avoid overlap that actually obscures required text. A source-raster adjacent free-dialogue candidate is valid only for the recorded required compact no-container role, writing-mode, geometry, and raster gates; candidate evidence does not authorize placement, and movement becomes authoritative only through the exact successful preview and commit path. Inspect its original crop, source and candidate bounds, direction/distance, rejected candidates, pixel/edge/boundary/background/contrast evidence, and explicitly reject movement that changes reading order or visual attribution. Legible decorative SFX must be translated with their detected typography preserved; skipped_difficult_sfx is permitted only for truly decorative, difficult-to-read sound effects. Reject any unjustified skip or any decorative-SFX classification of dialogue, caption, UI, container-associated, or general free text. Also reject OCR corruption, meaning drift, omissions, duplication, unnatural target-language spacing/grammar/wording, illegibility, and inappropriate typography or placement even when deterministic geometry checks pass.",
        line_break_review_policy: "Inspect each element's rendered_lines, not only aggregate line count, and reject visibly awkward word balance or short continuation orphans when the verified anchor has usable room.",
        required_judgments: [
            "OCR source text exactly matches the visible source and dialogue context",
            "translated meaning corresponds to the corrected source text",
            "target-language spacing, grammar, wording, register, and punctuation are natural",
            "reading order, segmentation, speaker/reaction proximity, and visual attribution preserve the original dialogue, including every adjacent free-dialogue anchor movement",
            "no source dialogue is omitted or spuriously duplicated",
            "translated text is visually readable and appropriately placed; every adjacent free-dialogue anchor remains close to its original speaker/reaction and passes the recorded source-pixel safety evidence",
            "every skipped_difficult_sfx item is truly decorative and difficult to OCR, and no required dialogue, caption, UI, or general free text was skipped",
        ],
        corrections,
        pages: &pages,
    };
    write_new(&manifest_path, &serde_json::to_vec_pretty(&manifest)?)?;
    Ok(ReviewBundle {
        directory: path_string(bundle_directory)?,
        manifest: path_string(&manifest_path)?,
        pages,
    })
}

pub(crate) fn write_page_artifacts(
    directory: &Path,
    page: ReviewPageInput,
) -> Result<ReviewBundlePage> {
    if directory.exists() {
        bail!(
            "visual-review page artifact directory already exists: {}",
            directory.display()
        );
    }
    std::fs::create_dir_all(directory).with_context(|| {
        format!(
            "failed to create visual-review page artifact directory {}",
            directory.display()
        )
    })?;
    let original_extension = media_extension(&page.original_media_type);
    let original = write_artifact(
        directory,
        &format!("original.{original_extension}"),
        &page.original_media_type,
        &page.original_bytes,
    )?;
    let rendered_preview = write_artifact(
        directory,
        "rendered.webp",
        "image/webp",
        &page.preview_bytes,
    )?;
    let semantic_bytes = serde_json::to_vec_pretty(&page.semantic_elements)?;
    let semantic_elements = write_artifact(
        directory,
        "semantic-elements.json",
        "application/json",
        &semantic_bytes,
    )?;
    Ok(ReviewBundlePage {
        page_id: page.page_id,
        label: page.label,
        original,
        rendered_preview,
        semantic_elements,
    })
}

fn write_artifact(
    directory: &Path,
    name: &str,
    media_type: &str,
    bytes: &[u8],
) -> Result<ReviewArtifact> {
    let path = directory.join(name);
    write_new(&path, bytes)?;
    Ok(ReviewArtifact {
        path: path_string(&path)?,
        media_type: media_type.to_owned(),
        byte_length: bytes.len(),
        blake3: blake3::hash(bytes).to_hex().to_string(),
    })
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("failed to create review artifact {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("failed to write review artifact {}", path.display()))
}

fn media_extension(media_type: &str) -> &'static str {
    match media_type {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        "image/bmp" => "bmp",
        "image/tiff" => "tiff",
        _ => "img",
    }
}

fn path_string(path: &Path) -> Result<String> {
    path.to_str()
        .context("review artifact path is not valid UTF-8")
        .map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_repair_plan(_revision: Revision) -> DeterministicRepairPlan {
        DeterministicRepairPlan {
            schema_version: crate::repair::REPAIR_PLAN_SCHEMA_VERSION,
            unresolved_failure_count: 0,
            blocking_failures: Vec::new(),
            terminal_diagnostic: None,
        }
    }

    fn page_artifacts(parent: &Path, original: &[u8], preview: &[u8]) -> ReviewBundlePage {
        write_page_artifacts(
            &parent.join("page-artifacts"),
            ReviewPageInput {
                page_id: EntityId::new(),
                label: "page.png".to_owned(),
                original_media_type: "image/png".to_owned(),
                original_bytes: original.to_vec(),
                preview_bytes: preview.to_vec(),
                semantic_elements: serde_json::json!({ "elements": [] }),
            },
        )
        .unwrap()
    }

    #[test]
    fn bundle_only_review_is_pending_and_contains_all_three_artifacts() {
        let parent = tempfile::tempdir().unwrap();
        let bundle = parent.path().join("review");
        let record = run_visual_review(
            &bundle,
            None,
            vec![page_artifacts(parent.path(), b"original", b"preview")],
            false,
            1,
            Revision::ZERO,
            Vec::new(),
            empty_repair_plan(Revision::ZERO),
        )
        .unwrap();
        assert_eq!(record.status, VisualReviewStatus::PendingAgentReview);
        assert!(!record.judge.required);
        assert!(!record.accepted());
        assert_eq!(record.bundle.pages.len(), 1);
        let page = &record.bundle.pages[0];
        assert!(Path::new(&page.original.path).is_file());
        assert!(Path::new(&page.rendered_preview.path).is_file());
        assert!(Path::new(&page.semantic_elements.path).is_file());
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&record.bundle.manifest).unwrap()).unwrap();
        assert_eq!(manifest["pages"].as_array().unwrap().len(), 1);
        assert_eq!(manifest["deterministic_acceptance_passed"], false);
        assert_eq!(
            manifest["deterministic_repair_plan"]["unresolved_failure_count"],
            0
        );
        assert!(
            manifest["purpose"]
                .as_str()
                .unwrap()
                .contains("not evidence")
        );
        assert!(
            manifest["line_break_review_policy"]
                .as_str()
                .unwrap()
                .contains("rendered_lines")
        );
    }

    #[test]
    fn refuses_to_overwrite_an_existing_bundle_directory() {
        let parent = tempfile::tempdir().unwrap();
        assert!(
            run_visual_review(
                parent.path(),
                None,
                Vec::new(),
                false,
                1,
                Revision::ZERO,
                Vec::new(),
                empty_repair_plan(Revision::ZERO),
            )
            .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn command_judge_decision_is_required_for_acceptance() {
        use std::os::unix::fs::PermissionsExt as _;

        let parent = tempfile::tempdir().unwrap();
        let judge = parent.path().join("judge");
        std::fs::write(
            &judge,
            "#!/bin/sh\nprintf '%s' '{\"accepted\":true,\"summary\":\"visually and semantically reviewed\",\"issues\":[],\"judgments\":{\"source_text_accurate\":true,\"translation_meaning_accurate\":true,\"target_language_natural\":true,\"reading_order_preserved\":true,\"content_complete_without_duplicates\":true,\"typography_layout_acceptable\":true,\"skipped_items_are_difficult_sfx_and_no_required_content_skipped\":true}}'\n",
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&judge).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&judge, permissions).unwrap();
        let record = run_visual_review(
            &parent.path().join("bundle"),
            Some(&judge),
            vec![page_artifacts(
                parent.path(),
                b"original source pixels",
                b"rendered preview pixels",
            )],
            true,
            1,
            Revision::ZERO,
            Vec::new(),
            empty_repair_plan(Revision::ZERO),
        )
        .unwrap();
        assert_eq!(record.status, VisualReviewStatus::Accepted);
        assert!(record.accepted());
        assert_eq!(record.bundle.pages.len(), 1);
        assert_eq!(record.bundle.pages[0].original.byte_length, 22);
        assert!(!record.bundle.pages[0].original.blake3.is_empty());
        assert_eq!(
            record.decision.unwrap().summary,
            "visually and semantically reviewed"
        );
    }

    #[cfg(unix)]
    #[test]
    fn attempt_four_semantic_and_linguistic_defects_cannot_pass_on_geometry() {
        use std::os::unix::fs::PermissionsExt as _;

        let parent = tempfile::tempdir().unwrap();
        let judge = parent.path().join("judge");
        std::fs::write(
            &judge,
            "#!/bin/sh\nprintf '%s' '{\"accepted\":true,\"summary\":\"geometry is readable but semantics fail\",\"issues\":[\"lower source OCR says O MAMA although dialogue context contradicts it\",\"Korean contains an unnatural spacing/grammar form: 너무뚱뚱해서\"],\"judgments\":{\"source_text_accurate\":false,\"translation_meaning_accurate\":false,\"target_language_natural\":false,\"reading_order_preserved\":true,\"content_complete_without_duplicates\":true,\"typography_layout_acceptable\":true,\"skipped_items_are_difficult_sfx_and_no_required_content_skipped\":true}}'\n",
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&judge).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&judge, permissions).unwrap();
        let record = run_visual_review(
            &parent.path().join("attempt-4"),
            Some(&judge),
            Vec::new(),
            true,
            4,
            Revision::ZERO,
            Vec::new(),
            empty_repair_plan(Revision::ZERO),
        )
        .unwrap();
        assert_eq!(record.status, VisualReviewStatus::Rejected);
        assert!(!record.accepted());
        assert_eq!(record.attempt, 4);
    }
}
