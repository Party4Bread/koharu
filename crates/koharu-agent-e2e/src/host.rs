use std::{
    collections::{BTreeMap, BTreeSet},
    io::Cursor,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use koharu_agent::{
    Control, Host, HostCompletion, HostTraceRecord, Invocation, Tool, ToolCall, ToolImageProvenance,
};
use koharu_config::Config;
use koharu_desktop::{ExportFormat, export_pages, rendered_preview};
use koharu_pipeline::{
    Committer, OcrModel, Operation, Pipeline, Progress, RunStatus, Scope, Stage, StageOutput,
    StopToken,
};
use koharu_rasterizer::Rasterizer;
use koharu_renderer::{LayerKind, RenderDiagnostic, Renderer};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, Authored, DetectionAnalysis, DetectionLabel,
    EntityId, FitsTo, FlowsIn, FontStyle, Generation, Geometry, Inside, LanguageTag, OcrAnalysis,
    PageDraft, PanelRegion, Patch, ProducerId, Region, RegionKind, RegionSpec, Revision, Session,
    Snapshot, SourceText, TextAlignment, TextDirection, TextLayout, TextLayoutKind, TextRegion,
    TextRole, Translation, Typography, Visibility, WritingMode,
};
use koharu_translator::{Language, ProvidersConfig};
use parking_lot::Mutex as SyncMutex;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::sync::{Mutex, OnceCell};
use unicode_segmentation::UnicodeSegmentation as _;

use crate::{
    OutputFormat,
    acceptance::{
        AcceptanceRecord, AcceptanceRejection, AcceptanceRejectionCode, ElementBounds,
        ElementGeometry, ElementPoint, ElementVisibility, FinalSceneElement, GlyphInkMask,
        HarnessConfiguration, LayoutAnchorMeasurement, LogicalDialogueGroupInspection,
        LogicalDialogueMemberInspection, LogicalDialogueMembershipInspection, PIXEL_HALF_DIAGONAL,
        PageInspection, ProjectInspection, ProjectState, QualityThresholds, SemanticText,
        TargetLayoutRelation, TargetRegionAssociation, TextElementInspection, TextSafeContainment,
        TextSafeRegion, evaluate,
    },
    free_dialogue::{
        ADJACENT_FREE_DIALOGUE_DETECTION_KIND, FREE_DIALOGUE_ANCHOR_SCHEMA_VERSION,
        FreeDialogueAnchorAssessment, FreeDialogueAnchorDecision, FreeDialogueAnchorInput,
        RASTER_FREE_DIALOGUE_DETECTOR, RASTER_FREE_DIALOGUE_DETECTOR_VERSION,
        assess_free_dialogue_anchor, validate_free_dialogue_anchor_decision,
        validate_source_bound_interjection_fallback,
    },
    page_translation::{
        RenderedPageReference, build_page_translation_dossier, build_source_evidence_dossier,
        page_reading_order_ordinals, render_page_debug_overlay, render_page_source_debug_overlay,
        write_page_debug_artifact, write_page_source_debug_artifact,
    },
    placement::is_actual_container_bound,
    repair::{
        AgentAction, CORRECTION_SCHEMA_VERSION, CorrectionRecord, DeterministicRepairFailure,
        DeterministicRepairPlan, RepairActionIdentity, RepairElementState, RepairField,
        RepairHistory, RepairStopDiagnostic, RepairText, RevisionEvidence,
        compact_translation_next_action, deterministic_repair_plan,
        infeasible_repair_stop_diagnostic, is_long_dialogue_clearance_failure,
        is_long_dialogue_target_anchor_failure, repair_stop_diagnostic,
    },
    review::{
        AGENT_VISUAL_SEMANTIC_REVIEW_SCHEMA_VERSION, AgentVisualSemanticReview,
        CompactedTranslationMemberEvidence, CompactedTranslationSemanticReview, ReviewPageInput,
        VisualReviewDecision, VisualReviewRecord, VisualReviewStatus, run_visual_review,
    },
    sfx::{
        DECORATIVE_SFX_DECISION_SCHEMA_VERSION, DECORATIVE_SFX_ROLE, DecorativeSfxDecision,
        DecorativeSfxDisposition, DecorativeSfxEvidence, FREE_TEXT_ROLE,
        MINIMUM_SFX_CLASSIFICATION_CONFIDENCE, SKIPPED_DIFFICULT_SFX_ROLE, SfxClassifier,
    },
    ui_panel::{
        DetectedPanelCandidate, MINIMUM_UI_PANEL_VERIFICATION_CONFIDENCE, RASTER_PANEL_DETECTOR,
        RASTER_PANEL_DETECTOR_VERSION, RasterPanelAssessment, RasterPanelEvidence, UI_TEXT_ROLE,
        UiPanelAnchorDecision, UiPanelClassifier, UiPanelEvidence, assess_source_raster_panel,
        raster_candidate_supports_ui_role, validate_raster_panel_evidence,
    },
};

#[derive(Clone, Debug, Serialize)]
pub struct PipelineTelemetry {
    pub schema_version: u32,
    pub status: PipelineRunTelemetryStatus,
    pub pages: Vec<PagePipelineTelemetry>,
    pub failure_stage: Option<Stage>,
    pub failure_kind: Option<String>,
    pub failure: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PageEvidenceArtifact {
    revision: Revision,
    page_id: EntityId,
    blake3: String,
    element_crops: BTreeMap<EntityId, SourceElementEvidence>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SourceElementEvidence {
    ordinal: usize,
    source_debug_label: String,
    crop_blake3: String,
}

/// A page's one-based position in the current `project.pages` order.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq)]
#[serde(transparent)]
struct PageOrdinal(NonZeroUsize);

impl PageOrdinal {
    #[cfg(test)]
    fn new(value: usize) -> Self {
        Self(NonZeroUsize::new(value).expect("page ordinals are one-based"))
    }

    fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PageTarget {
    ordinal: PageOrdinal,
    id: EntityId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScreeningEvidenceStage {
    InspectSourceEvidence,
    ViewPageSourceDebug,
    ClassifyDecorativeSfx,
}

impl ScreeningEvidenceStage {
    const fn as_str(self) -> &'static str {
        match self {
            Self::InspectSourceEvidence => "inspect_source_evidence",
            Self::ViewPageSourceDebug => "view_page_source_debug",
            Self::ClassifyDecorativeSfx => "classify_decorative_sfx",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SemanticReviewStage {
    InspectPageEvidence,
    SubmitVisualSemanticReview,
}

impl SemanticReviewStage {
    const fn as_str(self) -> &'static str {
        match self {
            Self::InspectPageEvidence => "inspect_page_evidence",
            Self::SubmitVisualSemanticReview => "submit_visual_semantic_review",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SemanticReviewObligation {
    page: PageTarget,
    required_stage: SemanticReviewStage,
}

impl SemanticReviewObligation {
    fn description(self) -> String {
        format!(
            "page ordinal {}, page ID {}, required stage {}",
            self.page.ordinal.get(),
            self.page.id,
            self.required_stage.as_str(),
        )
    }

    fn value(self) -> Value {
        json!({
            "page_ordinal": self.page.ordinal.get(),
            "page_id": self.page.id,
            "required_stage": self.required_stage.as_str(),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScreeningCandidateTarget {
    element_id: EntityId,
    original_ordinal: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ScreeningObligation {
    page: PageTarget,
    candidates: Vec<ScreeningCandidateTarget>,
    required_evidence_stage: ScreeningEvidenceStage,
}

impl ScreeningObligation {
    fn next_candidate(&self) -> ScreeningCandidateTarget {
        self.candidates[0]
    }

    fn description(&self) -> String {
        let candidate = self.next_candidate();
        format!(
            "page ordinal {}, page ID {}, element {}, original ordinal {}, required evidence stage {}",
            self.page.ordinal.get(),
            self.page.id,
            candidate.element_id,
            candidate.original_ordinal,
            self.required_evidence_stage.as_str(),
        )
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct PageSemanticEvidence {
    source_dossier: Option<PageEvidenceArtifact>,
    source_debug_artifact: Option<PageEvidenceArtifact>,
    dossier: Option<PageEvidenceArtifact>,
    debug_artifact: Option<PageEvidenceArtifact>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PageEvidenceKind {
    SourceDossier,
    SourceDebugArtifact,
    Dossier,
    DebugArtifact,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct PageSemanticEvidenceState {
    pages: BTreeMap<EntityId, PageSemanticEvidence>,
}

impl PageSemanticEvidenceState {
    fn record(&mut self, kind: PageEvidenceKind, artifact: PageEvidenceArtifact) {
        let page = self.pages.entry(artifact.page_id).or_default();
        match kind {
            PageEvidenceKind::SourceDossier => {
                page.source_dossier = Some(artifact);
                page.source_debug_artifact = None;
                page.dossier = None;
                page.debug_artifact = None;
            }
            PageEvidenceKind::SourceDebugArtifact => {
                page.source_debug_artifact = Some(artifact);
                page.dossier = None;
                page.debug_artifact = None;
            }
            PageEvidenceKind::Dossier => {
                page.dossier = Some(artifact);
                page.debug_artifact = None;
            }
            PageEvidenceKind::DebugArtifact => page.debug_artifact = Some(artifact),
        }
    }

    fn page(&self, page: EntityId) -> Option<&PageSemanticEvidence> {
        self.pages.get(&page)
    }
}

#[derive(Clone, Debug, Default)]
struct PageReviewState {
    content_revisions: BTreeMap<EntityId, Revision>,
    semantic_reviews: BTreeMap<EntityId, AgentVisualSemanticReview>,
}

impl PageReviewState {
    fn at_revision(pages: impl IntoIterator<Item = EntityId>, revision: Revision) -> Self {
        Self {
            content_revisions: pages.into_iter().map(|page| (page, revision)).collect(),
            semantic_reviews: BTreeMap::new(),
        }
    }

    fn record_review(&mut self, review: AgentVisualSemanticReview) {
        match self.semantic_reviews.entry(review.page_id) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(review);
            }
            std::collections::btree_map::Entry::Occupied(mut entry)
                if review.scene_revision >= entry.get().scene_revision =>
            {
                entry.insert(review);
            }
            std::collections::btree_map::Entry::Occupied(_) => {}
        }
    }

    fn record_mutation(&mut self, page: EntityId, revision: Revision) {
        self.content_revisions
            .entry(page)
            .and_modify(|current| *current = (*current).max(revision))
            .or_insert(revision);
    }

    fn current_review(&self, page: EntityId) -> Option<&AgentVisualSemanticReview> {
        let content_revision = self.content_revisions.get(&page)?;
        self.semantic_reviews
            .get(&page)
            .filter(|review| review.scene_revision >= *content_revision)
    }

    fn evidence_is_current(&self, evidence: &PageSemanticEvidenceState, page: EntityId) -> bool {
        let Some(content_revision) = self.content_revisions.get(&page) else {
            return false;
        };
        complete_page_evidence(evidence, page)
            .is_some_and(|(evidence_revision, _)| evidence_revision >= *content_revision)
    }

    fn current_reviews(
        &self,
        pages: impl IntoIterator<Item = EntityId>,
    ) -> Vec<AgentVisualSemanticReview> {
        pages
            .into_iter()
            .filter_map(|page| self.current_review(page).cloned())
            .collect()
    }
}

#[derive(Clone, Debug)]
struct CompactTranslationReviewRequirement {
    logical_group_id: EntityId,
    primary_render_element_id: EntityId,
    members: Vec<crate::repair::RepairLogicalDialogueMember>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineRunTelemetryStatus {
    Completed,
    Failed,
    Stopped,
}

#[derive(Clone, Debug, Serialize)]
pub struct PagePipelineTelemetry {
    pub page_id: EntityId,
    pub label: String,
    pub stages: Vec<StageTelemetry>,
    pub semantic_after: StageSemanticCounts,
    pub diagnosis: Vec<PipelineDiagnosis>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineDiagnosis {
    NoDetection,
    OcrEmpty,
    TranslationSkipped,
    TranslationFailed,
    TranslationOutputIncomplete,
    TypesettingFailed,
    ReadyForExport,
}

#[derive(Clone, Debug, Serialize)]
pub struct StageTelemetry {
    pub stage: Stage,
    pub status: StageTelemetryStatus,
    pub model: Option<String>,
    pub elapsed_ms: Option<u128>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageTelemetryStatus {
    NotRun,
    Loading,
    Running,
    Finished,
    NoOp,
    Skipped,
    Failed,
    Aborted,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct StageSemanticCounts {
    pub detected_text_elements: usize,
    pub required_source_elements: usize,
    pub skipped_difficult_sfx: usize,
    pub grouped_source_members: usize,
    pub logical_dialogue_groups: usize,
    pub required_render_units: usize,
    pub source_text_present: usize,
    pub source_text_nonempty: usize,
    pub translation_present: usize,
    pub translation_nonempty: usize,
    pub target_language_translations: usize,
    pub render_eligible_translations: usize,
    pub visible_translations: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum WorkflowPhase {
    SourceAnalysis,
    SourceEvidence,
    PageEvidence,
    DeterministicReview,
    DeterministicRepair,
    SemanticReview,
    SemanticRevision,
    Export,
    ReviewFailed,
    Completed,
}

impl WorkflowPhase {
    const fn as_str(self) -> &'static str {
        match self {
            Self::SourceAnalysis => "source_analysis",
            Self::SourceEvidence => "source_evidence",
            Self::PageEvidence => "page_evidence",
            Self::DeterministicReview => "deterministic_review",
            Self::DeterministicRepair => "deterministic_repair",
            Self::SemanticReview => "semantic_review",
            Self::SemanticRevision => "semantic_revision",
            Self::Export => "export",
            Self::ReviewFailed => "review_failed",
            Self::Completed => "completed",
        }
    }
}

#[derive(Clone, Debug)]
struct WorkflowSurfaceInputs {
    source_analysis_completed: bool,
    pipeline_completed: bool,
    source_evidence_ready: bool,
    decorative_sfx_dispositions_complete: bool,
    ui_panel_evidence_available: bool,
    page_evidence_revision: Option<Revision>,
    page_evidence_is_current: bool,
    current_page_review_accepted: Option<bool>,
    all_page_reviews_current: bool,
    semantic_review_stage: Option<SemanticReviewStage>,
    review: Option<VisualReviewRecord>,
    repair_tool: Option<&'static str>,
    repair_stopped: bool,
    exported: bool,
}

fn phase_tool_names(inputs: &WorkflowSurfaceInputs) -> (WorkflowPhase, Vec<&'static str>) {
    if inputs.exported {
        return (WorkflowPhase::Completed, Vec::new());
    }
    if !inputs.source_analysis_completed {
        return (WorkflowPhase::SourceAnalysis, vec!["run_source_analysis"]);
    }
    if !inputs.pipeline_completed {
        let mut names = vec!["inspect_source_evidence", "view_page_source_debug"];
        if inputs.source_evidence_ready {
            names.insert(2, "classify_decorative_sfx");
            if inputs.ui_panel_evidence_available {
                names.push("verify_ui_panel_anchor");
            }
        }
        if inputs.decorative_sfx_dispositions_complete {
            names.push("run_pipeline");
        }
        return (WorkflowPhase::SourceEvidence, names);
    }

    let evidence_tools = ["inspect_page_evidence"];
    let Some(review) = inputs.review.as_ref() else {
        let mut names = evidence_tools.to_vec();
        if inputs.page_evidence_is_current && inputs.page_evidence_revision.is_some() {
            names.push("revise_page_translation");
        }
        names.push("review_pages");
        return (WorkflowPhase::DeterministicReview, names);
    };
    if inputs.repair_stopped {
        return (WorkflowPhase::ReviewFailed, vec!["inspect_project"]);
    }
    let evidence_is_fresh =
        inputs.page_evidence_is_current && inputs.page_evidence_revision.is_some();
    if review.deterministic_repair_plan.unresolved_failure_count > 0 {
        let Some(repair_tool) = inputs.repair_tool else {
            return (WorkflowPhase::ReviewFailed, vec!["inspect_project"]);
        };
        if !matches!(
            repair_tool,
            "preview_compact_translation"
                | "preview_text_layout"
                | "commit_text_layout"
                | "commit_compact_translation"
                | "revise_page_translation"
        ) {
            return (WorkflowPhase::ReviewFailed, vec!["inspect_project"]);
        }
        let mut names = evidence_tools.to_vec();
        if evidence_is_fresh {
            names.push(repair_tool);
        }
        return (WorkflowPhase::DeterministicRepair, names);
    }
    if !review.deterministic_acceptance_passed {
        return (WorkflowPhase::ReviewFailed, vec!["inspect_project"]);
    }
    match review.status {
        VisualReviewStatus::PendingAgentReview => match inputs.semantic_review_stage {
            Some(SemanticReviewStage::SubmitVisualSemanticReview) => (
                WorkflowPhase::SemanticReview,
                vec![SemanticReviewStage::SubmitVisualSemanticReview.as_str()],
            ),
            Some(stage) => (WorkflowPhase::PageEvidence, vec![stage.as_str()]),
            None => (WorkflowPhase::ReviewFailed, vec!["inspect_project"]),
        },
        VisualReviewStatus::Accepted
            if inputs.all_page_reviews_current && !review.bundle.pages.is_empty() =>
        {
            (WorkflowPhase::Export, vec!["export_pages"])
        }
        VisualReviewStatus::Rejected
            if evidence_is_fresh && inputs.current_page_review_accepted == Some(false) =>
        {
            let mut names = evidence_tools.to_vec();
            names.push("revise_page_translation");
            (WorkflowPhase::SemanticRevision, names)
        }
        VisualReviewStatus::Rejected => (WorkflowPhase::PageEvidence, evidence_tools.to_vec()),
        VisualReviewStatus::Accepted | VisualReviewStatus::Failed => {
            (WorkflowPhase::ReviewFailed, vec!["inspect_project"])
        }
    }
}

fn actionable_completion<F>(
    phase: WorkflowPhase,
    tools: &[Tool],
    progress_marker: String,
    reason: F,
) -> Result<Option<HostCompletion>>
where
    F: FnOnce() -> Result<String>,
{
    if !matches!(
        phase,
        WorkflowPhase::DeterministicRepair | WorkflowPhase::SemanticRevision
    ) {
        return Ok(None);
    }
    Ok(Some(HostCompletion::Continue {
        phase: phase.as_str().to_owned(),
        exposed_tools: tools.iter().map(|tool| tool.name.clone()).collect(),
        reason: reason()?,
        progress_marker,
    }))
}

fn ensure_tool_available(phase: WorkflowPhase, tools: &[Tool], name: &str) -> Result<()> {
    if tools.iter().any(|tool| tool.name == name) {
        return Ok(());
    }
    if tool_definitions().iter().any(|tool| tool.name == name) {
        bail!(
            "Koharu harness tool {name} is unavailable during workflow phase {}; exposed tools: {}",
            phase.as_str(),
            tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    bail!("unknown Koharu harness tool {name}")
}

fn is_source_screening_tool(name: &str) -> bool {
    matches!(
        name,
        "inspect_source_evidence" | "view_page_source_debug" | "classify_decorative_sfx"
    )
}

fn validate_source_screening_tool_page(
    call: &ToolCall,
    obligation: &ScreeningObligation,
) -> Result<()> {
    let arguments: Value = serde_json::from_str(&call.arguments)
        .with_context(|| format!("invalid arguments for {}", call.name))?;
    let requested: PageOrdinal = serde_json::from_value(
        arguments
            .get("page_ordinal")
            .cloned()
            .with_context(|| format!("{} requires page_ordinal", call.name))?,
    )
    .with_context(|| format!("invalid page_ordinal for {}", call.name))?;
    if requested != obligation.page.ordinal {
        bail!(
            "{} is restricted to the next screening obligation: {}",
            call.name,
            obligation.description()
        );
    }
    Ok(())
}

fn is_semantic_review_tool(name: &str) -> bool {
    matches!(
        name,
        "inspect_page_evidence" | "submit_visual_semantic_review"
    )
}

fn validate_semantic_review_tool(
    call: &ToolCall,
    obligation: SemanticReviewObligation,
) -> Result<()> {
    if call.name != obligation.required_stage.as_str() {
        bail!(
            "{} is restricted by the next semantic-review obligation: {}",
            call.name,
            obligation.description()
        );
    }
    let arguments: Value = serde_json::from_str(&call.arguments)
        .with_context(|| format!("invalid arguments for {}", call.name))?;
    let requested: PageOrdinal = serde_json::from_value(
        arguments
            .get("page_ordinal")
            .cloned()
            .with_context(|| format!("{} requires page_ordinal", call.name))?,
    )
    .with_context(|| format!("invalid page_ordinal for {}", call.name))?;
    if requested != obligation.page.ordinal {
        bail!(
            "{} is restricted to the next semantic-review obligation: {}",
            call.name,
            obligation.description()
        );
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct MutableStageTelemetry {
    status: StageTelemetryStatus,
    model: Option<String>,
    elapsed_ms: Option<u128>,
}

#[derive(Clone)]
struct PreparedTextSafeLayoutRepair {
    preview: TextSafeLayoutRepairPreview,
    patch: Patch,
    content_id: EntityId,
    evidence: RevisionEvidence,
    before_state: RepairElementState,
    after_state: RepairElementState,
    reason: String,
}

#[derive(Clone)]
struct PreparedTextLayout {
    preview: TextLayoutPreview,
    patch: Patch,
    content_id: EntityId,
    evidence: RevisionEvidence,
    before_state: RepairElementState,
    after_state: RepairElementState,
    reason: String,
    changed_fields: Vec<&'static str>,
}

#[derive(Clone)]
struct PreparedCompactTranslation {
    preview: CompactTranslationPreview,
    patch: Patch,
    content_id: EntityId,
    evidence: RevisionEvidence,
    before_state: RepairElementState,
    after_state: RepairElementState,
    reason: String,
    member_ordinal_ids: Vec<crate::repair::RepairLogicalDialogueMember>,
}

#[derive(Clone, Debug)]
enum HostDeterministicRepair {
    TextLayout {
        element: EntityId,
        options: TextLayoutOptions,
        operation: &'static str,
    },
    TextSafeClearance {
        element: EntityId,
        inset_delta_px: TextSafeInsetDelta,
    },
}

impl HostDeterministicRepair {
    fn identity(&self) -> RepairActionIdentity {
        match self {
            Self::TextLayout {
                element, operation, ..
            } => RepairActionIdentity {
                operation,
                element: *element,
            },
            Self::TextSafeClearance { element, .. } => RepairActionIdentity {
                operation: "increase_text_safe_padding",
                element: *element,
            },
        }
    }
}

#[derive(Debug)]
enum HostDeterministicRepairOutcome {
    Committed {
        preview: Value,
        commit: Value,
    },
    Rejected {
        preview: Option<Value>,
        error: String,
    },
}

const HOST_DETERMINISTIC_CALL_PREFIX: &str = "host-deterministic-repair";

#[derive(Clone, Debug, Serialize)]
struct TextLayoutPreview {
    preview_id: String,
    operation: &'static str,
    base_revision: Revision,
    element_id: EntityId,
    options: TextLayoutOptions,
    metrics_before: TextLayoutPreviewMetrics,
    metrics_candidate: TextLayoutPreviewMetrics,
    deterministic_constraints_satisfied: bool,
    mutation_committed: bool,
}

#[derive(Clone, Debug, Serialize)]
struct TextLayoutPreviewMetrics {
    deterministic_element_accepted: bool,
    deterministic_rejection_codes: Vec<String>,
    finite_positive_layout: bool,
    line_count: Option<usize>,
    overflow: bool,
    rendered_font_size_px: Option<f64>,
    rendered_glyph_height_px: Option<f64>,
    maximum_reasonable_line_count: usize,
    target_layout_anchor: Option<LayoutAnchorMeasurement>,
    page_overflow_px: Option<f64>,
    text_safe_clearance: Option<TextSafeContainment>,
    renderer_diagnostics: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct CompactTranslationPreview {
    preview_id: String,
    operation: &'static str,
    base_revision: Revision,
    logical_group_id: EntityId,
    primary_render_element_id: EntityId,
    member_ordinal_ids: Vec<crate::repair::RepairLogicalDialogueMember>,
    current_visible_grapheme_units: usize,
    candidate_visible_grapheme_units: usize,
    metrics_before: TextLayoutPreviewMetrics,
    metrics_candidate: TextLayoutPreviewMetrics,
    deterministic_constraints_satisfied: bool,
    mutation_committed: bool,
}

#[derive(Clone, Debug, Serialize)]
struct TextSafeLayoutRepairPreview {
    preview_id: String,
    operation: &'static str,
    base_revision: Revision,
    element_id: EntityId,
    inset_delta_px: TextSafeInsetDelta,
    layout_bounds_before: ElementBounds,
    layout_bounds_candidate: ElementBounds,
    clearance_before: TextSafeContainment,
    clearance_candidate: TextSafeContainment,
    metrics_before: TextLayoutPreviewMetrics,
    metrics_candidate: TextLayoutPreviewMetrics,
    deterministic_diagnostics: Vec<AcceptanceRejection>,
    deterministic_constraints_satisfied: bool,
    committable: bool,
    mutation_committed: bool,
}

#[derive(Clone)]
pub(crate) struct DisposableProject(Arc<DisposableProjectInner>);

struct DisposableProjectInner {
    _root: TempDir,
    path: PathBuf,
    session: Arc<Mutex<Session>>,
    originals: Arc<[OriginalPage]>,
}

#[derive(Clone)]
struct ImportedPage {
    label: String,
    bytes: Arc<[u8]>,
    media_type: String,
    width: u32,
    height: u32,
}

#[derive(Clone)]
pub(crate) struct OriginalPage {
    pub label: String,
    pub bytes: Arc<[u8]>,
    pub media_type: String,
}

impl DisposableProject {
    pub(crate) async fn create(inputs: Vec<PathBuf>) -> Result<Self> {
        let root = tempfile::Builder::new()
            .prefix("koharu-agent-e2e-")
            .tempdir()?;
        let path = root.path().join("project.khrproj");
        let mut session = Session::create(&path)
            .await
            .with_context(|| format!("failed to create disposable project {}", path.display()))?;
        let pages = tokio::task::spawn_blocking(move || import_pages(inputs))
            .await
            .context("page import worker stopped unexpectedly")??;
        let originals = pages
            .iter()
            .map(|page| OriginalPage {
                label: page.label.clone(),
                bytes: Arc::clone(&page.bytes),
                media_type: page.media_type.clone(),
            })
            .collect::<Vec<_>>();
        let source = AssetRole::new("source")?;
        let patch = session.snapshot().patch(|edit| {
            for page in pages {
                let id = edit.add_page(
                    PageDraft::new(page.label, f64::from(page.width), f64::from(page.height)),
                    At::End,
                )?;
                edit.set_asset(
                    id,
                    &source,
                    AssetInput::new(
                        page.bytes,
                        page.media_type,
                        AssetMetadata {
                            width: Some(page.width),
                            height: Some(page.height),
                            attributes: Default::default(),
                        },
                    ),
                )?;
            }
            Ok(())
        })?;
        session.commit(patch).await?;
        Ok(Self(Arc::new(DisposableProjectInner {
            _root: root,
            path,
            session: Arc::new(Mutex::new(session)),
            originals: originals.into(),
        })))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0.path
    }

    pub(crate) fn session(&self) -> &Arc<Mutex<Session>> {
        &self.0.session
    }

    pub(crate) fn originals(&self) -> &[OriginalPage] {
        &self.0.originals
    }

    fn original_for_page<'a>(
        &'a self,
        snapshot: &Snapshot,
        page_id: EntityId,
    ) -> Result<&'a OriginalPage> {
        snapshot.page(page_id)?;
        let index = snapshot
            .pages()
            .position(|page| page.id() == page_id)
            .context("page is missing from snapshot order")?;
        self.originals()
            .get(index)
            .with_context(|| format!("original source page is missing for project page {page_id}"))
    }
}

fn import_pages(inputs: Vec<PathBuf>) -> Result<Vec<ImportedPage>> {
    inputs
        .into_iter()
        .map(|path| {
            let bytes = std::fs::read(&path)
                .with_context(|| format!("failed to read input page {}", path.display()))?;
            let format = image::guess_format(&bytes)
                .with_context(|| format!("unsupported input page {}", path.display()))?;
            let (width, height) = image::ImageReader::with_format(Cursor::new(&bytes), format)
                .into_dimensions()
                .with_context(|| format!("failed to read input page {}", path.display()))?;
            if width == 0 || height == 0 {
                bail!("input page is empty: {}", path.display());
            }
            Ok(ImportedPage {
                label: path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "page".to_owned()),
                bytes: Arc::from(bytes),
                media_type: format.to_mime_type().to_owned(),
                width,
                height,
            })
        })
        .collect()
}

#[derive(Clone)]
pub(crate) struct HarnessHost {
    project: DisposableProject,
    pipeline: Pipeline,
    renderer: Renderer,
    rasterizer: Arc<OnceCell<Arc<Rasterizer>>>,
    source_language: Language,
    target_language: Language,
    ocr_model: &'static str,
    output_directory: PathBuf,
    output_format: OutputFormat,
    quality_thresholds: QualityThresholds,
    configured_agent_model: Option<String>,
    review_bundle_directory: PathBuf,
    review_judge_command: Option<PathBuf>,
    outputs: Arc<SyncMutex<Vec<PathBuf>>>,
    source_analysis_completed: Arc<AtomicBool>,
    pipeline_completed: Arc<AtomicBool>,
    telemetry: Arc<SyncMutex<Option<PipelineTelemetry>>>,
    acceptance: Arc<SyncMutex<Option<AcceptanceRecord>>>,
    visual_review: Arc<SyncMutex<Option<VisualReviewRecord>>>,
    review_history: Arc<SyncMutex<Vec<VisualReviewRecord>>>,
    page_reviews: Arc<SyncMutex<PageReviewState>>,
    attempted_repair_actions: Arc<SyncMutex<BTreeSet<RepairActionIdentity>>>,
    repair_stop: Arc<SyncMutex<Option<RepairStopDiagnostic>>>,
    corrections: Arc<SyncMutex<Vec<CorrectionRecord>>>,
    last_explicit_observation: Arc<SyncMutex<Option<RevisionEvidence>>>,
    page_semantic_evidence: Arc<SyncMutex<PageSemanticEvidenceState>>,
    current_semantic_evidence_page: Arc<SyncMutex<Option<PageTarget>>>,
    decorative_sfx_decisions: Arc<SyncMutex<BTreeMap<EntityId, DecorativeSfxDecision>>>,
    pending_decorative_sfx_dispositions: Arc<SyncMutex<BTreeSet<EntityId>>>,
    ui_panel_anchors: Arc<SyncMutex<BTreeMap<EntityId, UiPanelAnchorDecision>>>,
    free_dialogue_anchors: Arc<SyncMutex<BTreeMap<EntityId, FreeDialogueAnchorDecision>>>,
    free_dialogue_anchor_assessments:
        Arc<SyncMutex<BTreeMap<EntityId, FreeDialogueAnchorAssessment>>>,
    raster_panel_evidence: Arc<SyncMutex<BTreeMap<EntityId, RasterPanelEvidence>>>,
    trace_records: Arc<SyncMutex<Vec<HostTraceRecord>>>,
    pending_text_safe_repairs: Arc<SyncMutex<BTreeMap<String, PreparedTextSafeLayoutRepair>>>,
    pending_text_layouts: Arc<SyncMutex<BTreeMap<String, PreparedTextLayout>>>,
    pending_compact_translations: Arc<SyncMutex<BTreeMap<String, PreparedCompactTranslation>>>,
}

impl HarnessHost {
    pub(crate) async fn create(
        inputs: Vec<PathBuf>,
        source_language: Language,
        target_language: Language,
        output_directory: PathBuf,
        output_format: OutputFormat,
        quality_thresholds: QualityThresholds,
        review_bundle_directory: PathBuf,
        review_judge_command: Option<PathBuf>,
    ) -> Result<Self> {
        let project = DisposableProject::create(inputs).await?;
        let pipeline_config = koharu_pipeline::PipelineConfig::load()?;
        let mut pipeline_config = pipeline_config.read()?.clone();
        configure_pipeline_for_harness(&mut pipeline_config, source_language, target_language);
        let ocr_model = match &pipeline_config.ocr {
            OcrModel::PaddleOcrVl1_6 => "paddleocr-vl-1.6",
            OcrModel::MangaOcr => "manga-ocr",
            OcrModel::BaberuOcr => "baberu-ocr",
            OcrModel::HayaiOcr => "hayai-ocr",
        };
        let pipeline = Pipeline::from_config(
            Config::memory(pipeline_config),
            ProvidersConfig::load()?,
            koharu_ml::device(false),
        )?;
        let configured_agent_model = koharu_agent::Config::load()?.read()?.model.clone();
        Ok(Self {
            project,
            pipeline,
            renderer: Renderer::new()?,
            rasterizer: Arc::new(OnceCell::new()),
            source_language,
            target_language,
            ocr_model,
            output_directory,
            output_format,
            quality_thresholds,
            configured_agent_model,
            review_bundle_directory,
            review_judge_command,
            outputs: Arc::new(SyncMutex::new(Vec::new())),
            source_analysis_completed: Arc::new(AtomicBool::new(false)),
            pipeline_completed: Arc::new(AtomicBool::new(false)),
            telemetry: Arc::new(SyncMutex::new(None)),
            acceptance: Arc::new(SyncMutex::new(None)),
            visual_review: Arc::new(SyncMutex::new(None)),
            review_history: Arc::new(SyncMutex::new(Vec::new())),
            page_reviews: Arc::new(SyncMutex::new(PageReviewState::default())),
            attempted_repair_actions: Arc::new(SyncMutex::new(BTreeSet::new())),
            repair_stop: Arc::new(SyncMutex::new(None)),
            corrections: Arc::new(SyncMutex::new(Vec::new())),
            last_explicit_observation: Arc::new(SyncMutex::new(None)),
            page_semantic_evidence: Arc::new(SyncMutex::new(PageSemanticEvidenceState::default())),
            current_semantic_evidence_page: Arc::new(SyncMutex::new(None)),
            decorative_sfx_decisions: Arc::new(SyncMutex::new(BTreeMap::new())),
            pending_decorative_sfx_dispositions: Arc::new(SyncMutex::new(BTreeSet::new())),
            ui_panel_anchors: Arc::new(SyncMutex::new(BTreeMap::new())),
            free_dialogue_anchors: Arc::new(SyncMutex::new(BTreeMap::new())),
            free_dialogue_anchor_assessments: Arc::new(SyncMutex::new(BTreeMap::new())),
            raster_panel_evidence: Arc::new(SyncMutex::new(BTreeMap::new())),
            trace_records: Arc::new(SyncMutex::new(Vec::new())),
            pending_text_safe_repairs: Arc::new(SyncMutex::new(BTreeMap::new())),
            pending_text_layouts: Arc::new(SyncMutex::new(BTreeMap::new())),
            pending_compact_translations: Arc::new(SyncMutex::new(BTreeMap::new())),
        })
    }

    pub(crate) fn project_path(&self) -> &Path {
        self.project.path()
    }

    pub(crate) fn output_paths(&self) -> Arc<SyncMutex<Vec<PathBuf>>> {
        Arc::clone(&self.outputs)
    }

    pub(crate) fn pipeline_completed(&self) -> bool {
        self.pipeline_completed.load(Ordering::Acquire)
    }

    pub(crate) fn pipeline_telemetry(&self) -> Option<PipelineTelemetry> {
        self.telemetry.lock().clone()
    }

    pub(crate) fn acceptance_record(&self) -> Option<AcceptanceRecord> {
        self.acceptance.lock().clone()
    }

    pub(crate) fn visual_review_record(&self) -> Option<VisualReviewRecord> {
        self.visual_review.lock().clone()
    }

    fn record_page_mutation(&self, page: EntityId, revision: Revision) {
        self.page_reviews.lock().record_mutation(page, revision);
    }

    fn reset_unresolved_semantic_evidence(&self, review: &VisualReviewRecord) {
        if review.judge.required
            || review.status != VisualReviewStatus::PendingAgentReview
            || !review.deterministic_acceptance_passed
            || review.deterministic_repair_plan.unresolved_failure_count > 0
        {
            return;
        }
        let page_reviews = self.page_reviews.lock();
        let accepted_pages = review
            .bundle
            .pages
            .iter()
            .filter(|page| {
                page_reviews
                    .current_review(page.page_id)
                    .is_some_and(|submitted| submitted.decision.accepted)
            })
            .map(|page| page.page_id)
            .collect::<BTreeSet<_>>();
        self.page_semantic_evidence
            .lock()
            .pages
            .retain(|page, _| accepted_pages.contains(page));
        if self
            .current_semantic_evidence_page
            .lock()
            .is_some_and(|page| !accepted_pages.contains(&page.id))
        {
            *self.current_semantic_evidence_page.lock() = None;
        }
    }

    fn next_screening_obligation(&self) -> Result<Option<ScreeningObligation>> {
        let pending = self.pending_decorative_sfx_dispositions.lock().clone();
        if pending.is_empty() {
            return Ok(None);
        }
        let snapshot = self
            .project
            .session()
            .try_lock()
            .context("source-phase tool selection requires an idle project session")?
            .snapshot();
        screening_obligation(&snapshot, &pending, &self.page_semantic_evidence.lock()).map(Some)
    }

    fn next_semantic_review_obligation(&self) -> Result<Option<SemanticReviewObligation>> {
        let Some(review) = self.visual_review_record() else {
            return Ok(None);
        };
        if review.judge.required
            || review.status != VisualReviewStatus::PendingAgentReview
            || !review.deterministic_acceptance_passed
            || review.deterministic_repair_plan.unresolved_failure_count > 0
        {
            return Ok(None);
        }
        let snapshot = self
            .project
            .session()
            .try_lock()
            .context("semantic-review tool selection requires an idle project session")?
            .snapshot();
        semantic_review_obligation(
            &snapshot,
            &review,
            &self.page_reviews.lock(),
            &self.page_semantic_evidence.lock(),
        )
    }

    fn workflow_surface(&self) -> (WorkflowPhase, Vec<Tool>) {
        let evidence = self.page_semantic_evidence.lock().clone();
        let screening_obligation = self
            .next_screening_obligation()
            .expect("pending source screening state must resolve against the current snapshot");
        let current_evidence_page = *self.current_semantic_evidence_page.lock();
        let page_evidence_revision = current_evidence_page
            .and_then(|page| complete_page_evidence(&evidence, page.id).map(|value| value.0));
        let page_reviews = self.page_reviews.lock().clone();
        let page_evidence_is_current = current_evidence_page
            .is_some_and(|page| page_reviews.evidence_is_current(&evidence, page.id));
        let current_page_review_accepted = current_evidence_page
            .and_then(|page| page_reviews.current_review(page.id))
            .map(|review| review.decision.accepted);
        let source_evidence_ready = screening_obligation.as_ref().map_or_else(
            || {
                current_evidence_page
                    .is_some_and(|page| matching_source_evidence(&evidence, page.id))
            },
            |obligation| {
                obligation.required_evidence_stage == ScreeningEvidenceStage::ClassifyDecorativeSfx
            },
        );
        let review = self.visual_review_record();
        let semantic_review_obligation = self
            .next_semantic_review_obligation()
            .expect("pending semantic-review state must resolve against the current snapshot");
        let all_page_reviews_current = review.as_ref().is_some_and(|review| {
            review.judge.required
                || review
                    .bundle
                    .pages
                    .iter()
                    .all(|page| page_reviews.current_review(page.page_id).is_some())
        });
        let repair_tool = review
            .as_ref()
            .and_then(|review| self.active_repair_tool(review));
        let bound_ui_panels = self
            .ui_panel_anchors
            .lock()
            .values()
            .map(|decision| decision.panel.region_id)
            .collect::<std::collections::BTreeSet<_>>();
        let ui_panel_evidence_available = self
            .raster_panel_evidence
            .lock()
            .keys()
            .any(|panel| !bound_ui_panels.contains(panel));
        let pending_screening_candidates = self.pending_decorative_sfx_dispositions.lock().len();
        let inputs = WorkflowSurfaceInputs {
            source_analysis_completed: self.source_analysis_completed.load(Ordering::Acquire),
            pipeline_completed: self.pipeline_completed(),
            source_evidence_ready,
            decorative_sfx_dispositions_complete: pending_screening_candidates == 0,
            ui_panel_evidence_available,
            page_evidence_revision,
            page_evidence_is_current,
            current_page_review_accepted,
            all_page_reviews_current,
            semantic_review_stage: semantic_review_obligation
                .map(|obligation| obligation.required_stage),
            review: review.clone(),
            repair_tool: repair_tool.as_ref().map(|value| value.name),
            repair_stopped: self.repair_stop.lock().is_some(),
            exported: !self.outputs.lock().is_empty(),
        };
        let (phase, names) = phase_tool_names(&inputs);
        let tools = names
            .into_iter()
            .map(|name| {
                let mut tool = tool_definition(name).clone();
                if let Some(repair) = repair_tool.as_ref()
                    && repair.name == name
                {
                    constrain_repair_tool(&mut tool, repair);
                }
                if name == "classify_decorative_sfx" {
                    tool.description.push_str(&format!(
                        " Current pending screening candidates: {pending_screening_candidates}."
                    ));
                }
                if matches!(
                    name,
                    "inspect_source_evidence"
                        | "view_page_source_debug"
                        | "classify_decorative_sfx"
                ) && let Some(obligation) = screening_obligation.as_ref()
                {
                    constrain_usize_property(
                        &mut tool.parameters,
                        "page_ordinal",
                        obligation.page.ordinal.get(),
                    );
                    tool.description.push_str(&format!(
                        " Next screening obligation: {}. Pending target-page candidates: {}.",
                        obligation.description(),
                        obligation
                            .candidates
                            .iter()
                            .map(|candidate| format!(
                                "original ordinal {} element {}",
                                candidate.original_ordinal, candidate.element_id
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                    if name == "classify_decorative_sfx" {
                        constrain_object_keyword(
                            &mut tool.parameters,
                            "/$defs/DecorativeSfxClassification/properties/element",
                            "enum",
                            json!(
                                obligation
                                    .candidates
                                    .iter()
                                    .map(|candidate| candidate.element_id.to_string())
                                    .collect::<Vec<_>>()
                            ),
                        );
                        constrain_object_keyword(
                            &mut tool.parameters,
                            "/properties/decisions",
                            "maxItems",
                            json!(obligation.candidates.len()),
                        );
                    }
                }
                if name == "submit_visual_semantic_review"
                    && let Some(obligation) = semantic_review_obligation
                {
                    let page_evidence = evidence
                        .page(obligation.page.id)
                        .expect("submit obligation requires page evidence");
                    constrain_usize_property(
                        &mut tool.parameters,
                        "page_ordinal",
                        obligation.page.ordinal.get(),
                    );
                    constrain_u64_property(
                        &mut tool.parameters,
                        "scene_revision",
                        review
                            .as_ref()
                            .expect("semantic-review obligation requires a review")
                            .scene_revision
                            .get(),
                    );
                    for (property, artifact) in [
                        (
                            "source_evidence_dossier_blake3",
                            page_evidence.source_dossier.as_ref(),
                        ),
                        (
                            "source_debug_artifact_blake3",
                            page_evidence.source_debug_artifact.as_ref(),
                        ),
                        ("dossier_blake3", page_evidence.dossier.as_ref()),
                        ("debug_artifact_blake3", page_evidence.debug_artifact.as_ref()),
                    ] {
                        constrain_string_property(
                            &mut tool.parameters,
                            property,
                            artifact
                                .expect("submit obligation requires complete page evidence")
                                .blake3
                                .clone(),
                        );
                    }
                    tool.description.push_str(
                        " The page ordinal, scene revision, and all four evidence digests are exact schema constants for direct submission.",
                    );
                }
                if let Some(obligation) = semantic_review_obligation
                    && name == obligation.required_stage.as_str()
                {
                    constrain_usize_property(
                        &mut tool.parameters,
                        "page_ordinal",
                        obligation.page.ordinal.get(),
                    );
                    tool.description.push_str(&format!(
                        " Exact next semantic-review obligation: {}.",
                        obligation.description()
                    ));
                }
                if name == "revise_page_translation"
                    && let Some((revision, page)) = current_evidence_page.and_then(|page| {
                        complete_page_evidence(&evidence, page.id).map(|value| (value.0, page))
                    })
                {
                    constrain_usize_property(
                        &mut tool.parameters,
                        "page_ordinal",
                        page.ordinal.get(),
                    );
                    constrain_pointer_const(
                        &mut tool.parameters,
                        "/$defs/PageTranslationEvidenceReference/properties/scene_revision",
                        json!(revision.get()),
                    );
                }
                tool
            })
            .collect();
        (phase, tools)
    }

    fn continuation_reason(&self, phase: WorkflowPhase) -> Result<String> {
        match phase {
            WorkflowPhase::DeterministicRepair => {
                let review = self
                    .visual_review_record()
                    .context("deterministic repair phase is missing its review record")?;
                let failure = review
                    .deterministic_repair_plan
                    .blocking_failures
                    .first()
                    .context("deterministic repair phase has no blocking failure")?;
                let code = serde_json::to_value(failure.code)?
                    .as_str()
                    .context("deterministic repair failure code did not serialize as text")?
                    .to_owned();
                let target = failure
                    .element_id
                    .map_or_else(|| "project".to_owned(), |element| element.to_string());
                Ok(format!(
                    "deterministic review recorded {code} for {target} at revision {}",
                    failure.required_evidence_revision
                ))
            }
            WorkflowPhase::SemanticRevision => {
                let page = self
                    .current_semantic_evidence_page
                    .lock()
                    .context("semantic revision phase has no current evidence page")?;
                let page_reviews = self.page_reviews.lock();
                let decision = &page_reviews
                    .current_review(page.id)
                    .context("semantic revision phase has no current rejected page review")?
                    .decision;
                let issues = decision.issues.join("; ");
                Ok(format!(
                    "semantic review rejected page ordinal {}: {}; issues: {issues}",
                    page.ordinal.get(),
                    decision.summary
                ))
            }
            _ => bail!(
                "workflow phase {} is not eligible for host continuation",
                phase.as_str()
            ),
        }
    }

    fn continuation_progress_marker(
        &self,
        phase: WorkflowPhase,
        scene_revision: Revision,
    ) -> String {
        let review = self.visual_review_record();
        json!({
            "scene_revision": scene_revision.get(),
            "workflow_phase": phase.as_str(),
            "deterministic_review": review.as_ref().map(|review| json!({
                "scene_revision": review.scene_revision.get(),
                "status": review.status,
                "unresolved_failure_count": review
                    .deterministic_repair_plan
                    .unresolved_failure_count,
            })),
        })
        .to_string()
    }

    fn active_repair_tool(&self, review: &VisualReviewRecord) -> Option<RepairToolConstraint> {
        if let Some(commit) = pending_text_layout_commit(review, &self.pending_text_layouts.lock())
        {
            return Some(commit);
        }
        if let Some(commit) = pending_compact_translation_commit(
            review,
            &self.pending_compact_translations.lock(),
            &self.page_semantic_evidence.lock(),
        ) {
            return Some(commit);
        }
        let failure = review.deterministic_repair_plan.blocking_failures.first()?;
        if let Some(action) = failure.next_action.as_ref().filter(|action| {
            matches!(
                action.tool,
                "preview_compact_translation" | "preview_text_layout"
            )
        }) {
            return Some(RepairToolConstraint {
                name: action.tool,
                operation: Some(action.operation),
                element: Some(action.element),
                logical_group: failure.logical_group_id,
                page: None,
                revision: Some(action.required_evidence_revision),
                preview_id: None,
                allowed_fields: failure.allowed_repair_fields.clone(),
            });
        }
        failure
            .element_id
            .filter(|_| {
                failure.allowed_repair_fields.iter().any(|field| {
                    matches!(
                        field,
                        RepairField::SourceText | RepairField::TranslationText
                    )
                })
            })
            .map(|element| RepairToolConstraint {
                name: "revise_page_translation",
                operation: None,
                element: Some(element),
                logical_group: failure.logical_group_id,
                page: (*self.current_semantic_evidence_page.lock()).and_then(|page| {
                    complete_page_evidence(&self.page_semantic_evidence.lock(), page.id)
                        .map(|_| page.ordinal)
                }),
                revision: Some(failure.required_evidence_revision),
                preview_id: None,
                allowed_fields: failure.allowed_repair_fields.clone(),
            })
    }

    async fn rasterizer(&self) -> Result<Arc<Rasterizer>> {
        self.rasterizer
            .get_or_try_init(|| async {
                let rasterizer = tokio::task::spawn_blocking(Rasterizer::new)
                    .await
                    .context("native rasterizer worker stopped unexpectedly")??;
                Ok::<_, anyhow::Error>(Arc::new(rasterizer))
            })
            .await
            .cloned()
    }

    async fn project_context(&self) -> Result<Value> {
        let mut context = serde_json::to_value(self.inspect_project().await?)?;
        if let Some(pages) = context
            .pointer_mut("/project/pages")
            .and_then(Value::as_array_mut)
        {
            for (index, page) in pages.iter_mut().enumerate() {
                if let Some(page) = page.as_object_mut() {
                    page.insert("page_ordinal".to_owned(), json!(index + 1));
                }
            }
        }
        Ok(context)
    }

    async fn inspect_project(&self) -> Result<ProjectInspection> {
        let snapshot = self.project.session().lock().await.snapshot();
        self.inspect_snapshot(snapshot).await
    }

    async fn inspect_snapshot(&self, snapshot: Snapshot) -> Result<ProjectInspection> {
        let decorative_sfx_decisions = self.decorative_sfx_decisions.lock().clone();
        let ui_panel_anchors = self.ui_panel_anchors.lock().clone();
        let free_dialogue_anchors = self.free_dialogue_anchors.lock().clone();
        let free_dialogue_anchor_assessments = self.free_dialogue_anchor_assessments.lock().clone();
        let raster_panel_evidence = self.raster_panel_evidence.lock().clone();
        let mut pages = Vec::new();
        for page in snapshot.pages() {
            let id = page.id();
            let value = page.page()?;
            let rendered = self.renderer.render(&snapshot, id).await;
            let render_error = rendered.as_ref().err().map(|error| format!("{error:#}"));
            let mut text_elements = Vec::new();
            let mut logical_dialogue_groups = Vec::new();
            let detected_panel_candidates =
                detected_panel_candidates(&snapshot, id, &raster_panel_evidence)?;
            if let Some(group) = page.text_group()? {
                let layers = group.text_layers()?.collect::<Vec<_>>();
                for layer in &layers {
                    let content = layer.content()?;
                    let Some(dialogue) = content.logical_dialogue()? else {
                        continue;
                    };
                    let members = dialogue
                        .members
                        .iter()
                        .map(|member| {
                            let source = snapshot
                                .component::<SourceText>(member.content_id)?
                                .map_or_else(String::new, |source| source.text.value);
                            Ok::<_, anyhow::Error>(LogicalDialogueMemberInspection {
                                ordinal: member.ordinal,
                                element_id: member.element_id,
                                content_id: member.content_id,
                                source_region_id: member.source_region_id,
                                source_text: source,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    logical_dialogue_groups.push(LogicalDialogueGroupInspection {
                        group_id: content.id(),
                        primary_render_element_id: dialogue.members[0].element_id,
                        target_region_id: dialogue.target_region_id,
                        logical_source_text: members
                            .iter()
                            .map(|member| member.source_text.as_str())
                            .collect::<Vec<_>>()
                            .join("\n"),
                        members,
                    });
                }
                for layer in layers {
                    let layer_id = layer.id();
                    let content = layer.content()?;
                    let source_region = content.source_region()?;
                    let source_region_id = source_region.map(|region| region.id());
                    let source_region_kind = source_region
                        .map(|region| region.region().map(|value| value.kind.as_str().to_owned()))
                        .transpose()?;
                    let detected = match source_region {
                        Some(region) => {
                            region.region()?.kind == TextRegion::kind()
                                && region.detection()?.is_some()
                        }
                        None => false,
                    };
                    let text_role = content.role()?.map(|role| role.role);
                    let decorative_sfx = decorative_sfx_decisions.get(&layer_id).cloned();
                    let verified_ui_panel_anchor = ui_panel_anchors.get(&layer_id).cloned();
                    let verified_free_dialogue_anchor =
                        free_dialogue_anchors.get(&layer_id).cloned();
                    let free_dialogue_anchor_assessment = source_region_id
                        .and_then(|source| free_dialogue_anchor_assessments.get(&source).cloned());
                    let source = content.source()?.map(|source| SemanticText {
                        text: source.text.value,
                        language: source.language.map(|language| language.to_string()),
                    });
                    let translation = content.translation()?.map(|translation| SemanticText {
                        text: translation.text.value,
                        language: translation.language.map(|language| language.to_string()),
                    });
                    let local = layer.visibility()?.unwrap_or(Visibility {
                        origin: koharu_scene::Origin::User,
                        visible: true,
                        opacity: 1.0,
                    });
                    let (effective_visible, effective_opacity) =
                        effective_visibility(&snapshot, layer_id)?;
                    let source_geometry = source_region
                        .map(|region| region.geometry().map(element_geometry))
                        .transpose()?;
                    let source_analysis = match source_region {
                        Some(region) => snapshot.component::<OcrAnalysis>(region.id())?,
                        None => None,
                    };
                    let source_writing_mode =
                        source_analysis.and_then(|analysis| match analysis.direction {
                            TextDirection::Horizontal => Some(WritingMode::Horizontal),
                            TextDirection::Vertical => Some(WritingMode::Vertical),
                            TextDirection::Auto => None,
                        });
                    let balloon_target = layer.balloon_target()?;
                    let fit_target = layer.fit_target()?;
                    let verified_panel_target = verified_ui_panel_anchor
                        .as_ref()
                        .and_then(|decision| {
                            detected_panel_candidates
                                .iter()
                                .find(|panel| panel.region_id == decision.panel.region_id)
                        })
                        .map(|panel| snapshot.analysis_region(panel.region_id))
                        .transpose()?;
                    let verified_free_dialogue_target = verified_free_dialogue_anchor
                        .as_ref()
                        .map(|decision| snapshot.analysis_region(decision.target_region_id))
                        .transpose()?
                        .filter(|target| fit_target.is_some_and(|fit| fit.id() == target.id()));
                    let text_safe_target = verified_free_dialogue_target
                        .or(verified_panel_target)
                        .or(balloon_target);
                    let text_safe_region = text_safe_target
                        .or(source_region)
                        .map(|region| {
                            let layout_relation = if balloon_target
                                .is_some_and(|target| target.id() == region.id())
                            {
                                Some(TargetLayoutRelation::FlowsIn)
                            } else if fit_target.is_some_and(|target| target.id() == region.id()) {
                                Some(TargetLayoutRelation::FitsTo)
                            } else {
                                None
                            };
                            let association = layout_relation.map(|layout_relation| {
                                let source_inside_target_relation = source_region_id
                                    .map(|source| {
                                        snapshot
                                            .relations_from_as::<Inside>(source)
                                            .any(|relation| relation.value().target == region.id())
                                    })
                                    .unwrap_or(false);
                                TargetRegionAssociation {
                                    layout_relation,
                                    source_inside_target_relation,
                                }
                            });
                            Ok::<_, anyhow::Error>(TextSafeRegion {
                                id: region.id(),
                                kind: region.region()?.kind.as_str().to_owned(),
                                geometry: element_geometry(region.geometry()?),
                                association,
                            })
                        })
                        .transpose()?;
                    let typography = layer.typography()?;
                    let layout_kind = layer.layout()?.kind;
                    let authored_layout_geometry = snapshot
                        .component::<Geometry>(layer_id)?
                        .map(element_geometry);
                    let mut final_scene = rendered
                        .as_ref()
                        .ok()
                        .and_then(|frame| frame.layer(layer_id))
                        .map(|rendered_layer| {
                            let presentation = rendered_layer.presentation();
                            match rendered_layer.kind() {
                                LayerKind::Text(metadata) => {
                                    let glyph_bounds = render_bounds(metadata.rendered_bounds);
                                    let layout_bounds = render_bounds(metadata.layout_bounds);
                                    let font_size_px = f64::from(metadata.font_size)
                                        .is_finite()
                                        .then_some(f64::from(metadata.font_size));
                                    let geometry_visible = glyph_bounds.is_some_and(|bounds| {
                                        bounds_intersect_page(bounds, value.width, value.height)
                                    });
                                    let mut diagnostics = rendered
                                        .as_ref()
                                        .ok()
                                        .map(|frame| text_diagnostics(frame, layer_id))
                                        .unwrap_or_default();
                                    if glyph_bounds.is_none() {
                                        diagnostics.push("non_finite_glyph_bounds".to_owned());
                                    }
                                    if layout_bounds.is_none() {
                                        diagnostics.push("non_finite_layout_bounds".to_owned());
                                    }
                                    if font_size_px.is_none() {
                                        diagnostics.push("non_finite_font_size".to_owned());
                                    }
                                    FinalSceneElement {
                                        eligible: true,
                                        visible: presentation.visible
                                            && presentation.opacity > 0.0
                                            && geometry_visible,
                                        opacity: presentation.opacity,
                                        geometry_visible,
                                        glyph_bounds,
                                        layout_bounds,
                                        font_size_px,
                                        line_count: Some(metadata.line_count),
                                        rendered_lines: metadata.rendered_lines.clone(),
                                        diagnostics,
                                        glyph_ink: None,
                                    }
                                }
                                LayerKind::Image(_) => FinalSceneElement::default(),
                            }
                        })
                        .unwrap_or_default();
                    if final_scene.eligible
                        && let Ok(frame) = &rendered
                    {
                        match self.rasterized_glyph_ink(frame, layer_id).await {
                            Ok(mask) => final_scene.glyph_ink = mask,
                            Err(error) => final_scene
                                .diagnostics
                                .push(format!("glyph_ink_rasterization_failed: {error:#}")),
                        }
                    }
                    text_elements.push(TextElementInspection {
                        id: layer_id,
                        content_id: content.id(),
                        source_region_id,
                        source_region_kind,
                        detected,
                        required: detected
                            && decorative_sfx
                                .as_ref()
                                .is_none_or(DecorativeSfxDecision::requires_translation),
                        text_role,
                        decorative_sfx,
                        logical_dialogue_memberships: logical_dialogue_groups
                            .iter()
                            .filter_map(|dialogue| {
                                dialogue
                                    .members
                                    .iter()
                                    .find(|member| member.element_id == layer_id)
                                    .map(|member| LogicalDialogueMembershipInspection {
                                        group_id: dialogue.group_id,
                                        primary_render_element_id: dialogue
                                            .primary_render_element_id,
                                        target_region_id: dialogue.target_region_id,
                                        member_ordinal: member.ordinal,
                                    })
                            })
                            .collect(),
                        source,
                        translation,
                        source_writing_mode,
                        visibility: ElementVisibility {
                            local_visible: local.visible,
                            local_opacity: local.opacity,
                            effective_visible,
                            effective_opacity,
                        },
                        source_geometry,
                        text_safe_region,
                        verified_ui_panel_anchor,
                        verified_free_dialogue_anchor,
                        free_dialogue_anchor_assessment,
                        typography,
                        layout_kind,
                        authored_layout_geometry,
                        final_scene,
                    });
                }
            }
            pages.push(PageInspection {
                id,
                label: value.label,
                width: value.width,
                height: value.height,
                text_elements,
                detected_panel_candidates,
                logical_dialogue_groups,
                render_error,
            });
        }
        let review_history = self.review_history.lock();
        let active_deterministic_plan = review_history.last().and_then(|review| {
            (review.deterministic_repair_plan.unresolved_failure_count > 0)
                .then(|| review.deterministic_repair_plan.clone())
        });
        let completed_review_attempts = review_history.len() as u32;
        drop(review_history);
        Ok(ProjectInspection {
            project: ProjectState {
                kind: "isolated_disposable",
                revision: snapshot.revision(),
                pages,
            },
            configuration: HarnessConfiguration {
                source_language: self.source_language.tag().to_owned(),
                target_language: self.target_language.tag().to_owned(),
                ocr_model: self.ocr_model.to_owned(),
                required_output_directory: path_string(&self.output_directory)?,
                required_export_format: self.output_format.as_str().to_owned(),
                quality_thresholds: self.quality_thresholds.clone(),
                review_bundle_directory: path_string(&self.review_bundle_directory)?,
                external_visual_judge_configured: self.review_judge_command.is_some(),
            },
            repair_history: RepairHistory {
                completed_review_attempts,
                attempted_actions: self
                    .attempted_repair_actions
                    .lock()
                    .iter()
                    .cloned()
                    .collect(),
                active_deterministic_plan,
                stop_diagnostic: self.repair_stop.lock().clone(),
                correction_actions: self.corrections.lock().clone(),
            },
        })
    }

    async fn rasterized_glyph_ink(
        &self,
        frame: &koharu_renderer::Frame,
        element: EntityId,
    ) -> Result<Option<GlyphInkMask>> {
        let Some(cropped) = frame.cropped(element)? else {
            return Ok(None);
        };
        let raster_frame = cropped.raster_frame()?;
        let rasterizer = self.rasterizer().await?;
        let raster = tokio::task::spawn_blocking(move || {
            rasterizer.rasterize(&raster_frame, koharu_rasterizer::RasterOptions::default())
        })
        .await
        .context("glyph-ink rasterizer worker stopped unexpectedly")??;
        let (width, height) = raster.image.dimensions();
        let alpha = raster
            .image
            .into_raw()
            .chunks_exact(4)
            .map(|pixel| pixel[3])
            .collect();
        Ok(Some(GlyphInkMask {
            left: raster.left,
            top: raster.top,
            width,
            height,
            alpha,
        }))
    }

    async fn record_acceptance(&self) -> Result<AcceptanceRecord> {
        let inspection = self.inspect_project().await?;
        let record = evaluate(&inspection);
        *self.acceptance.lock() = Some(record.clone());
        self.trace_records.lock().push(HostTraceRecord::new(
            "acceptance",
            serde_json::to_value(&record)?,
        ));
        Ok(record)
    }

    async fn record_visual_review(&self) -> Result<VisualReviewRecord> {
        if let Some(diagnostic) = self.repair_stop.lock().clone() {
            bail!(
                "deterministic repair loop stopped: {}",
                serde_json::to_string(&diagnostic)?
            );
        }
        if let Some(record) = self.visual_review_record() {
            return Ok(record);
        }
        let review_attempt = self.review_history.lock().len() as u32 + 1;
        let acceptance = match self.acceptance_record() {
            Some(record) => record,
            None => self.record_acceptance().await?,
        };
        let inspection = self.inspect_project().await?;
        let originals = self.project.originals();
        if originals.len() != inspection.project.pages.len() {
            bail!(
                "review input mismatch: {} originals for {} inspected pages",
                originals.len(),
                inspection.project.pages.len()
            );
        }
        let snapshot = self.project.session().lock().await.snapshot();
        let scene_revision = snapshot.revision();
        let deterministic_repair_plan = deterministic_repair_plan(
            &acceptance.rejection_reasons,
            scene_revision,
            Some(&inspection),
        );
        let previous_review = self.review_history.lock().last().cloned();
        if let Some(previous_review) = previous_review
            && previous_review.scene_revision != scene_revision
            && let Some(diagnostic) = repair_stop_diagnostic(
                &previous_review.deterministic_repair_plan,
                &deterministic_repair_plan,
            )
        {
            *self.repair_stop.lock() = Some(diagnostic.clone());
            self.trace_records.lock().push(HostTraceRecord::new(
                "repair_stopped",
                serde_json::to_value(&diagnostic)?,
            ));
            bail!(
                "deterministic repair made no progress: the same operation and target remained active after mutation; diagnostic: {}",
                serde_json::to_string(&diagnostic)?
            );
        }
        let corrections = self.corrections.lock().clone();
        let rasterizer = self.rasterizer().await?;
        let mut pages = Vec::with_capacity(originals.len());
        for (index, (original, page)) in originals.iter().zip(&inspection.project.pages).enumerate()
        {
            let preview_bytes =
                rendered_preview(&self.renderer, rasterizer.clone(), &snapshot, page.id)
                    .await
                    .with_context(|| {
                        format!("failed to render review preview for {}", page.label)
                    })?;
            pages.push(ReviewPageInput {
                page_id: page.id,
                label: original.label.clone(),
                original_media_type: original.media_type.clone(),
                original_bytes: original.bytes.to_vec(),
                preview_bytes,
                semantic_elements: json!({
                    "schema_version": 6,
                    "source_language": acceptance.source_language,
                    "target_language": acceptance.target_language,
                    "thresholds": acceptance.thresholds,
                    "page": page,
                    "acceptance": acceptance.pages.get(index),
                    "corrections": corrections
                        .iter()
                        .filter(|correction| page.text_elements.iter().any(|element| {
                            element.id == correction.element_id
                        }))
                        .collect::<Vec<_>>(),
                }),
            });
        }
        let bundle_directory = self
            .review_bundle_directory
            .join(format!("attempt-{review_attempt}"));
        let judge_command = self.review_judge_command.clone();
        let deterministic_acceptance_passed = acceptance.accepted;
        let mut record = tokio::task::spawn_blocking(move || {
            run_visual_review(
                &bundle_directory,
                judge_command.as_deref(),
                pages,
                deterministic_acceptance_passed,
                review_attempt,
                scene_revision,
                corrections,
                deterministic_repair_plan,
            )
        })
        .await
        .context("visual-review worker stopped unexpectedly")??;
        if !record.judge.required {
            record.agent_reviews = self
                .page_reviews
                .lock()
                .current_reviews(record.bundle.pages.iter().map(|page| page.page_id));
            refresh_agent_visual_semantic_review_status(&mut record);
        }
        *self.visual_review.lock() = Some(record.clone());
        self.review_history.lock().push(record.clone());
        let mut review_trace = serde_json::to_value(&record)?;
        if let Some(object) = review_trace.as_object_mut() {
            object.insert(
                "free_dialogue_anchor_review_evidence".to_owned(),
                json!(inspection
                    .project
                    .pages
                    .iter()
                    .flat_map(|page| &page.text_elements)
                    .filter(|element| element.free_dialogue_anchor_assessment.is_some())
                    .map(|element| json!({
                        "element_id": element.id,
                        "source_ocr": element.source,
                        "verified_anchor": element.verified_free_dialogue_anchor,
                        "assessment": element.free_dialogue_anchor_assessment,
                        "required_in_loop_review": "inspect proximity to the original speaker/reaction and reject changed reading order or visual attribution",
                    }))
                    .collect::<Vec<_>>()),
            );
        }
        self.trace_records
            .lock()
            .push(HostTraceRecord::new("visual_review", review_trace));
        if let Some(diagnostic) =
            infeasible_repair_stop_diagnostic(&record.deterministic_repair_plan)
        {
            *self.repair_stop.lock() = Some(diagnostic.clone());
            self.trace_records.lock().push(HostTraceRecord::new(
                "repair_stopped",
                serde_json::to_value(&diagnostic)?,
            ));
        }
        Ok(record)
    }

    async fn record_review_and_execute_host_repairs(&self) -> Result<VisualReviewRecord> {
        let mut review = self.record_visual_review().await?;
        loop {
            if self.repair_stop.lock().is_some() {
                return Ok(review);
            }
            let Some(failure) = review
                .deterministic_repair_plan
                .blocking_failures
                .first()
                .cloned()
            else {
                return Ok(review);
            };
            let repair = match host_deterministic_repair(&failure) {
                Ok(Some(repair)) => repair,
                Ok(None) => return Ok(review),
                Err(error) => {
                    self.stop_host_deterministic_repair(&review, None, error.to_string())?;
                    return Ok(review);
                }
            };
            let identity = repair.identity();
            if !record_repair_action_attempt(
                &mut self.attempted_repair_actions.lock(),
                identity.clone(),
            ) {
                self.stop_host_deterministic_repair(
                    &review,
                    None,
                    format!(
                        "host deterministic repair action {} for element {} was already attempted without progress",
                        identity.operation, identity.element
                    ),
                )?;
                return Ok(review);
            }
            let outcome = self.execute_host_deterministic_repair(&repair).await;
            match outcome {
                HostDeterministicRepairOutcome::Committed { preview, commit } => {
                    self.trace_records.lock().push(HostTraceRecord::new(
                        "host_deterministic_repair_committed",
                        json!({
                            "schema_version": 1,
                            "reviewed_revision": review.scene_revision,
                            "repair": host_deterministic_repair_trace(&repair),
                            "preview": preview,
                            "commit": commit,
                        }),
                    ));
                    review = self.record_visual_review().await?;
                }
                HostDeterministicRepairOutcome::Rejected { preview, error } => {
                    self.trace_records.lock().push(HostTraceRecord::new(
                        "host_deterministic_repair_rejected",
                        json!({
                            "schema_version": 1,
                            "reviewed_revision": review.scene_revision,
                            "repair": host_deterministic_repair_trace(&repair),
                            "preview": preview,
                            "error": error,
                        }),
                    ));
                    if let Some(updated) = self.visual_review_record()
                        && updated.scene_revision == review.scene_revision
                        && updated
                            .deterministic_repair_plan
                            .blocking_failures
                            .first()
                            .and_then(|failure| failure.next_action.as_ref())
                            .is_some_and(|action| action.tool == "preview_compact_translation")
                    {
                        return Ok(updated);
                    }
                    self.stop_host_deterministic_repair(&review, preview, error)?;
                    return Ok(review);
                }
            }
        }
    }

    async fn execute_host_deterministic_repair(
        &self,
        repair: &HostDeterministicRepair,
    ) -> HostDeterministicRepairOutcome {
        match repair {
            HostDeterministicRepair::TextLayout {
                element,
                options,
                operation,
            } => {
                let preview_call = ToolCall {
                    call_id: format!("{HOST_DETERMINISTIC_CALL_PREFIX}-preview-{element}"),
                    name: "preview_text_layout".to_owned(),
                    arguments: json!({
                        "element": element,
                        "options": options,
                        "reason": format!("host-owned deterministic repair: {operation}"),
                    })
                    .to_string(),
                };
                let preview = match self.preview_text_layout(&preview_call).await {
                    Ok(preview) => preview,
                    Err(error) => {
                        return HostDeterministicRepairOutcome::Rejected {
                            preview: None,
                            error: format!("{error:#}"),
                        };
                    }
                };
                let Some(preview_id) = preview
                    .value
                    .pointer("/preview/preview_id")
                    .and_then(Value::as_str)
                else {
                    return HostDeterministicRepairOutcome::Rejected {
                        preview: Some(preview.value),
                        error: "host text-layout preview did not issue an exact preview ID"
                            .to_owned(),
                    };
                };
                let commit_call = ToolCall {
                    call_id: format!("{HOST_DETERMINISTIC_CALL_PREFIX}-commit-{element}"),
                    name: "commit_text_layout".to_owned(),
                    arguments: json!({ "preview_id": preview_id }).to_string(),
                };
                match self.commit_text_layout(&commit_call).await {
                    Ok(commit) => HostDeterministicRepairOutcome::Committed {
                        preview: preview.value,
                        commit: commit.value,
                    },
                    Err(error) => HostDeterministicRepairOutcome::Rejected {
                        preview: Some(preview.value),
                        error: format!("{error:#}"),
                    },
                }
            }
            HostDeterministicRepair::TextSafeClearance {
                element,
                inset_delta_px,
            } => {
                let preview_call = ToolCall {
                    call_id: format!("{HOST_DETERMINISTIC_CALL_PREFIX}-preview-{element}"),
                    name: "preview_increase_text_safe_padding".to_owned(),
                    arguments: json!({
                        "element": element,
                        "inset_delta_px": inset_delta_px,
                        "reason": "host-owned deterministic text-safe clearance repair",
                    })
                    .to_string(),
                };
                let preview = match self.preview_text_safe_layout_repair(&preview_call).await {
                    Ok(preview) => preview,
                    Err(error) => {
                        return HostDeterministicRepairOutcome::Rejected {
                            preview: None,
                            error: format!("{error:#}"),
                        };
                    }
                };
                let Some(preview_id) = preview
                    .value
                    .pointer("/preview/preview_id")
                    .and_then(Value::as_str)
                    .filter(|_| {
                        preview
                            .value
                            .pointer("/preview/committable")
                            .and_then(Value::as_bool)
                            == Some(true)
                    })
                else {
                    let error = preview
                        .value
                        .pointer("/rejection/message")
                        .and_then(Value::as_str)
                        .unwrap_or("host text-safe preview failed global deterministic validation")
                        .to_owned();
                    return HostDeterministicRepairOutcome::Rejected {
                        preview: Some(preview.value),
                        error,
                    };
                };
                let commit_call = ToolCall {
                    call_id: format!("{HOST_DETERMINISTIC_CALL_PREFIX}-commit-{element}"),
                    name: "commit_text_safe_layout_repair".to_owned(),
                    arguments: json!({ "preview_id": preview_id }).to_string(),
                };
                match self.commit_text_safe_layout_repair(&commit_call).await {
                    Ok(commit) => HostDeterministicRepairOutcome::Committed {
                        preview: preview.value,
                        commit: commit.value,
                    },
                    Err(error) => HostDeterministicRepairOutcome::Rejected {
                        preview: Some(preview.value),
                        error: format!("{error:#}"),
                    },
                }
            }
        }
    }

    fn stop_host_deterministic_repair(
        &self,
        review: &VisualReviewRecord,
        preview: Option<Value>,
        error: String,
    ) -> Result<()> {
        let failure = review
            .deterministic_repair_plan
            .blocking_failures
            .first()
            .cloned();
        let diagnostic = RepairStopDiagnostic {
            schema_version: crate::repair::REPAIR_PLAN_SCHEMA_VERSION,
            reason: "no allowed host deterministic repair passed global validation",
            previous_evidence_revision: review.scene_revision,
            reviewed_revision: review.scene_revision,
            previous_unresolved_failure_count: review
                .deterministic_repair_plan
                .unresolved_failure_count,
            current_unresolved_failure_count: review
                .deterministic_repair_plan
                .unresolved_failure_count,
            first_unresolved_failure: failure,
            infeasible_required_layout: review
                .deterministic_repair_plan
                .terminal_diagnostic
                .clone(),
        };
        *self.repair_stop.lock() = Some(diagnostic.clone());
        self.trace_records.lock().push(HostTraceRecord::new(
            "repair_stopped",
            json!({
                "diagnostic": diagnostic,
                "preview": preview,
                "error": error,
            }),
        ));
        Ok(())
    }

    async fn submit_visual_semantic_review(&self, call: &ToolCall) -> Result<Invocation> {
        if !self.pipeline_completed() {
            bail!("the complete Koharu pipeline must finish before visual/semantic review");
        }
        let arguments: SubmitVisualSemanticReview = arguments(call)?;
        let snapshot = self.project.session().lock().await.snapshot();
        let page = resolve_page_ordinal(&snapshot, arguments.page_ordinal)?;
        let page_id = page.id;
        let current_revision = snapshot.revision();
        let requested_revision = Revision::new(arguments.scene_revision);
        let mut review = self.visual_review_record().context(
            "submit_visual_semantic_review requires review_pages for the current revision",
        )?;
        if review.scene_revision != current_revision {
            bail!(
                "review_pages evaluated revision {}, not current revision {current_revision}",
                review.scene_revision
            );
        }
        if !self
            .page_reviews
            .lock()
            .evidence_is_current(&self.page_semantic_evidence.lock(), page_id)
        {
            bail!(
                "agent visual/semantic review evidence is stale for page {page_id} at current project revision {current_revision}"
            );
        }
        let compact_requirements =
            compact_translation_review_requirements(&snapshot, page_id, &self.corrections.lock())?;
        let submitted = validate_agent_visual_semantic_review(
            &review,
            &self.page_semantic_evidence.lock(),
            requested_revision,
            page_id,
            &compact_requirements,
            &arguments,
        )?;
        record_agent_visual_semantic_review(&mut review, submitted.clone());
        self.page_reviews.lock().record_review(submitted.clone());
        let mut history = self.review_history.lock();
        let historical = history
            .last_mut()
            .context("visual-review history is missing the current review")?;
        if historical.attempt != review.attempt
            || historical.scene_revision != review.scene_revision
        {
            bail!("visual-review history does not match the current review");
        }
        *historical = review.clone();
        drop(history);
        *self.visual_review.lock() = Some(review.clone());
        self.trace_records.lock().push(HostTraceRecord::new(
            "agent_visual_semantic_review",
            serde_json::to_value(&submitted)?,
        ));
        let export_precondition_satisfied = review.accepted();
        let next_action = if export_precondition_satisfied {
            "export_pages after deterministic acceptance"
        } else if review.rejected() {
            "repair the concrete issues, refresh bundled page evidence, then run review_pages again"
        } else {
            "inspect_page_evidence and submit_visual_semantic_review for each remaining page"
        };
        Invocation::read(json!({
            "agent_review": submitted,
            "visual_review": review,
            "export_precondition_satisfied": export_precondition_satisfied,
            "next_action": next_action,
        }))
    }

    async fn inspect_page_evidence(&self, call: &ToolCall) -> Result<Invocation> {
        if !self.pipeline_completed() {
            bail!("the complete Koharu pipeline must finish before page evidence inspection");
        }
        let arguments: InspectPageEvidence = arguments(call)?;
        let snapshot = self.project.session().lock().await.snapshot();
        let page = resolve_page_ordinal(&snapshot, arguments.page_ordinal)?;
        let revision = snapshot.revision();
        let evidence_before = self.page_semantic_evidence.lock().clone();
        let current_page_before = *self.current_semantic_evidence_page.lock();
        let observation_before = self.last_explicit_observation.lock().clone();
        let trace_count_before = self.trace_records.lock().len();
        let artifact_call = |suffix: &str, name: &str| ToolCall {
            call_id: format!("{}-{suffix}", call.call_id),
            name: name.to_owned(),
            arguments: json!({ "page_ordinal": arguments.page_ordinal.get() }).to_string(),
        };

        let bundled = async {
            let source = self
                .inspect_source_evidence(&artifact_call("source", "inspect_source_evidence"))
                .await?;
            let source_debug = self
                .view_page_source_debug(&artifact_call("source-debug", "view_page_source_debug"))
                .await?;
            let translated = self
                .review_page_translation(&artifact_call("translated", "review_page_translation"))
                .await?;
            let rendered_debug = self
                .view_page_debug(&artifact_call("rendered-debug", "view_page_debug"))
                .await?;

            let source_evidence = source
                .value
                .get("source_evidence")
                .cloned()
                .context("source evidence inspection omitted its dossier")?;
            let source_evidence_dossier_blake3 = source
                .value
                .get("source_evidence_dossier_blake3")
                .and_then(Value::as_str)
                .context("source evidence inspection omitted its dossier digest")?
                .to_owned();
            let source_debug_artifact_blake3 = source_debug
                .value
                .get("blake3")
                .and_then(Value::as_str)
                .context("source debug inspection omitted its artifact digest")?
                .to_owned();
            let dossier_blake3 = translated
                .value
                .get("dossier_blake3")
                .and_then(Value::as_str)
                .context("translated dossier inspection omitted its digest")?
                .to_owned();
            let debug_artifact_blake3 = rendered_debug
                .value
                .get("blake3")
                .and_then(Value::as_str)
                .context("rendered debug inspection omitted its artifact digest")?
                .to_owned();
            let mut translated_dossier = translated
                .value
                .get("dossier")
                .cloned()
                .context("translated dossier inspection omitted its dossier")?;
            translated_dossier
                .as_object_mut()
                .context("translated dossier must be an object")?
                .insert(
                    "accepted_predecessor_context".to_owned(),
                    translated
                        .value
                        .get("accepted_predecessor_context")
                        .cloned()
                        .context("translated dossier omitted predecessor context")?,
                );

            ensure!(
                complete_page_evidence(&self.page_semantic_evidence.lock(), page.id)
                    == Some((revision, page.id)),
                "bundled page evidence did not bind one exact page revision"
            );
            *self.last_explicit_observation.lock() =
                Some(RevisionEvidence::PageTranslationVisualEvidence {
                    revision,
                    page_id: page.id,
                    source_evidence_dossier_blake3: source_evidence_dossier_blake3.clone(),
                    source_debug_artifact_blake3: source_debug_artifact_blake3.clone(),
                    dossier_blake3: dossier_blake3.clone(),
                    debug_artifact_blake3: debug_artifact_blake3.clone(),
                });

            let mut invocation = Invocation::read(json!({
                "page_ordinal": page.ordinal.get(),
                "page_id": page.id,
                "scene_revision": revision,
                "source_evidence": source_evidence,
                "source_debug_artifact": source_debug.value,
                "translated_dossier": translated_dossier,
                "rendered_debug_artifact": rendered_debug.value,
                "source_evidence_dossier_blake3": source_evidence_dossier_blake3,
                "source_debug_artifact_blake3": source_debug_artifact_blake3,
                "dossier_blake3": dossier_blake3,
                "debug_artifact_blake3": debug_artifact_blake3,
                "next_action": "submit_visual_semantic_review",
            }))?;
            invocation.images = source.images;
            invocation.images.extend(source_debug.images);
            invocation.images.extend(translated.images);
            invocation.images.extend(rendered_debug.images);
            Ok(invocation)
        }
        .await;

        if bundled.is_err() {
            *self.page_semantic_evidence.lock() = evidence_before;
            *self.current_semantic_evidence_page.lock() = current_page_before;
            *self.last_explicit_observation.lock() = observation_before;
            self.trace_records.lock().truncate(trace_count_before);
        }
        bundled
    }

    async fn review_page_translation(&self, call: &ToolCall) -> Result<Invocation> {
        if !self.pipeline_completed() {
            bail!("the complete Koharu pipeline must finish before page translation review");
        }
        let arguments: ReviewPageTranslation = arguments(call)?;
        let snapshot = self.project.session().lock().await.snapshot();
        let page_target = resolve_page_ordinal(&snapshot, arguments.page_ordinal)?;
        let page_id = page_target.id;
        let revision = snapshot.revision();
        snapshot.page(page_id)?;
        let inspection = self.inspect_snapshot(snapshot.clone()).await?;
        let acceptance = evaluate(&inspection);
        let page = inspection
            .project
            .pages
            .iter()
            .find(|page| page.id == page_id)
            .context("reviewed page is missing from project inspection")?;
        let page_acceptance = acceptance.pages.iter().find(|page| page.page_id == page_id);
        let corrections = self.corrections.lock().clone();
        let page_reviews = self.page_reviews.lock().clone();
        let accepted_predecessor_context = inspection
            .project
            .pages
            .iter()
            .take(page_target.ordinal.get() - 1)
            .enumerate()
            .filter_map(|(index, predecessor)| {
                let accepted = page_reviews
                    .current_review(predecessor.id)
                    .filter(|review| review.decision.accepted)?;
                let source_translation_pairs = page_reading_order_ordinals(predecessor)
                    .into_iter()
                    .filter_map(|(ordinal, element_id)| {
                        let element = predecessor
                            .text_elements
                            .iter()
                            .find(|element| element.id == element_id)?;
                        let original_ocr = corrections
                            .iter()
                            .find(|record| record.element_id == element_id)
                            .and_then(|record| record.original_ocr.as_ref())
                            .map(|text| SemanticText {
                                text: text.text.clone(),
                                language: text.language.clone(),
                            })
                            .or_else(|| element.source.clone());
                        Some(json!({
                            "ordinal": ordinal,
                            "element_id": element_id,
                            "original_ocr": original_ocr,
                            "current_source": element.source,
                            "current_translation": element.translation,
                        }))
                    })
                    .collect::<Vec<_>>();
                Some(json!({
                    "page_ordinal": index + 1,
                    "page_id": predecessor.id,
                    "page_label": predecessor.label,
                    "review_accepted": true,
                    "review_scene_revision": accepted.scene_revision,
                    "review_summary": accepted.decision.summary,
                    "source_translation_pairs": source_translation_pairs,
                }))
            })
            .collect::<Vec<_>>();
        let rendered =
            rendered_preview(&self.renderer, self.rasterizer().await?, &snapshot, page_id).await?;
        let rendered_page = RenderedPageReference {
            media_type: "image/webp",
            byte_length: rendered.len(),
            blake3: blake3::hash(&rendered).to_hex().to_string(),
        };
        let dossier = build_page_translation_dossier(
            revision,
            page,
            page_acceptance,
            self.source_language.tag(),
            self.target_language.tag(),
            |element, current| {
                corrections
                    .iter()
                    .find(|record| record.element_id == element)
                    .and_then(|record| record.original_ocr.as_ref())
                    .map(|text| SemanticText {
                        text: text.text.clone(),
                        language: text.language.clone(),
                    })
                    .or_else(|| current.clone())
            },
            rendered_page,
        );
        let dossier_value = serde_json::to_value(&dossier)?;
        let dossier_bytes = serde_json::to_vec(&dossier_value)?;
        let dossier_digest = blake3::hash(&dossier_bytes).to_hex().to_string();
        *self.last_explicit_observation.lock() = Some(RevisionEvidence::PageTranslationReview {
            revision,
            page_id,
            dossier_digest: dossier_digest.clone(),
        });
        self.page_semantic_evidence.lock().record(
            PageEvidenceKind::Dossier,
            PageEvidenceArtifact {
                revision,
                page_id,
                blake3: dossier_digest.clone(),
                element_crops: BTreeMap::new(),
            },
        );
        *self.current_semantic_evidence_page.lock() = Some(page_target);
        self.trace_records.lock().push(HostTraceRecord::new(
            "page_translation_review",
            json!({
                "schema_version": 3,
                "scene_revision": revision,
                "page_id": page_id,
                "dossier_blake3": dossier_digest,
                "rendered_page": dossier.rendered_page,
                "element_count": dossier.elements.len(),
                "target_layout_anchors": dossier.elements.iter().map(|element| json!({
                    "element_id": element.element_id,
                    "text_role": element.text_role,
                    "verified_ui_panel_anchor": element.verified_ui_panel_anchor,
                    "verified_free_dialogue_anchor": element.verified_free_dialogue_anchor,
                    "free_dialogue_anchor_assessment": element.free_dialogue_anchor_assessment,
                    "target_layout_anchor": element.layout.target_layout_anchor,
                })).collect::<Vec<_>>(),
            }),
        ));
        Invocation::read(json!({
            "dossier": dossier_value,
            "dossier_blake3": dossier_digest,
            "accepted_predecessor_context": accepted_predecessor_context,
            "next_actions": ["view_page_debug", "revise_page_translation", "preview_text_layout", "review_pages"],
        }))
    }

    async fn inspect_source_evidence(&self, call: &ToolCall) -> Result<Invocation> {
        if !self.source_analysis_completed.load(Ordering::Acquire) && !self.pipeline_completed() {
            bail!(
                "source detection and OCR analysis must finish before source evidence inspection"
            );
        }
        let arguments: InspectSourceEvidence = arguments(call)?;
        let snapshot = self.project.session().lock().await.snapshot();
        let page_target = resolve_page_ordinal(&snapshot, arguments.page_ordinal)?;
        let page_id = page_target.id;
        let revision = snapshot.revision();
        snapshot.page(page_id)?;
        let original = self.project.original_for_page(&snapshot, page_id)?.clone();
        let inspection = self.inspect_snapshot(snapshot.clone()).await?;
        let page = inspection
            .project
            .pages
            .iter()
            .find(|page| page.id == page_id)
            .context("source evidence page is missing from project inspection")?;
        let corrections = self.corrections.lock().clone();
        let mut ocr_confidences = BTreeMap::new();
        for region in page
            .text_elements
            .iter()
            .filter_map(|element| element.source_region_id)
        {
            let confidence = snapshot
                .component::<OcrAnalysis>(region)?
                .and_then(|analysis| analysis.confidence);
            ocr_confidences.insert(region, confidence);
        }
        let dossier = build_source_evidence_dossier(
            &self.review_bundle_directory,
            revision,
            page,
            &original.bytes,
            &original.media_type,
            |element, current| {
                corrections
                    .iter()
                    .find(|record| record.element_id == element)
                    .and_then(|record| record.original_ocr.as_ref())
                    .map(|text| SemanticText {
                        text: text.text.clone(),
                        language: text.language.clone(),
                    })
                    .or_else(|| current.clone())
            },
            |region| region.and_then(|id| ocr_confidences.get(&id).copied().flatten()),
        )?;
        let mut dossier_value = serde_json::to_value(&dossier)?;
        let pending = self.pending_decorative_sfx_dispositions.lock().clone();
        let pending_on_page = dossier
            .elements
            .iter()
            .filter(|element| pending.contains(&element.element_id))
            .map(|element| {
                json!({
                    "element_id": element.element_id,
                    "original_ordinal": element.ordinal,
                })
            })
            .collect::<Vec<_>>();
        if let Some(object) = dossier_value.as_object_mut() {
            object.insert(
                "pending_decorative_sfx_dispositions".to_owned(),
                Value::Array(pending_on_page),
            );
            if let Some(elements) = object.get_mut("elements").and_then(Value::as_array_mut) {
                for element in elements {
                    let is_pending = element
                        .get("element_id")
                        .and_then(Value::as_str)
                        .and_then(|id| entity(id).ok())
                        .is_some_and(|id| pending.contains(&id));
                    if let Some(element) = element.as_object_mut() {
                        element.insert(
                            "pending_decorative_sfx_disposition".to_owned(),
                            json!(is_pending),
                        );
                    }
                }
            }
        }
        let dossier_digest = blake3::hash(&serde_json::to_vec(&dossier_value)?)
            .to_hex()
            .to_string();
        *self.last_explicit_observation.lock() = Some(RevisionEvidence::SourceEvidenceInspection {
            revision,
            page_id,
            dossier_digest: dossier_digest.clone(),
        });
        self.page_semantic_evidence.lock().record(
            PageEvidenceKind::SourceDossier,
            PageEvidenceArtifact {
                revision,
                page_id,
                blake3: dossier_digest.clone(),
                element_crops: dossier
                    .elements
                    .iter()
                    .map(|element| {
                        (
                            element.element_id,
                            SourceElementEvidence {
                                ordinal: element.ordinal,
                                source_debug_label: element.source_debug_label.clone(),
                                crop_blake3: element.original_crop.blake3.clone(),
                            },
                        )
                    })
                    .collect(),
            },
        );
        *self.current_semantic_evidence_page.lock() = Some(page_target);
        self.trace_records.lock().push(HostTraceRecord::new(
            "source_evidence_inspection",
            json!({
                "schema_version": 2,
                "scene_revision": revision,
                "page_id": page_id,
                "source_evidence_dossier_blake3": dossier_digest,
                "full_page_original": dossier.full_page_original,
                "detected_panel_candidates": dossier.detected_panel_candidates,
                "elements": dossier.elements.iter().map(|element| json!({
                    "ordinal": element.ordinal,
                    "element_id": element.element_id,
                    "source_debug_label": element.source_debug_label,
                    "review_state": element.review_state,
                    "text_role": element.text_role,
                    "verified_ui_panel_anchor": element.verified_ui_panel_anchor,
                    "verified_free_dialogue_anchor": element.verified_free_dialogue_anchor,
                    "free_dialogue_anchor_assessment": element.free_dialogue_anchor_assessment,
                    "decorative_sfx": element.decorative_sfx,
                    "original_crop": element.original_crop,
                })).collect::<Vec<_>>(),
            }),
        ));
        let mut invocation = Invocation::read(json!({
            "source_evidence": dossier_value,
            "source_evidence_dossier_blake3": dossier_digest,
            "next_action": "view_page_source_debug",
        }))?;
        invocation = invocation.with_image(
            format!("Original full page {} ({page_id})", original.label),
            format!(
                "data:{};base64,{}",
                original.media_type,
                STANDARD.encode(&original.bytes)
            ),
            ToolImageProvenance::ContentHash {
                algorithm: "blake3",
                digest: dossier.full_page_original.blake3.clone(),
                media_type: original.media_type,
                byte_length: original.bytes.len(),
            },
        );
        for element in &dossier.elements {
            let bytes = std::fs::read(&element.original_crop.path).with_context(|| {
                format!(
                    "failed to read source crop artifact {}",
                    element.original_crop.path
                )
            })?;
            invocation = invocation.with_image(
                format!(
                    "Original source crop {} element {}",
                    element.source_debug_label, element.element_id
                ),
                format!("data:image/png;base64,{}", STANDARD.encode(&bytes)),
                ToolImageProvenance::ContentHash {
                    algorithm: "blake3",
                    digest: element.original_crop.blake3.clone(),
                    media_type: element.original_crop.media_type.clone(),
                    byte_length: element.original_crop.byte_length,
                },
            );
        }
        Ok(invocation)
    }

    async fn view_page_source_debug(&self, call: &ToolCall) -> Result<Invocation> {
        if !self.source_analysis_completed.load(Ordering::Acquire) && !self.pipeline_completed() {
            bail!("source detection and OCR analysis must finish before source debug review");
        }
        let arguments: ViewPageSourceDebug = arguments(call)?;
        let snapshot = self.project.session().lock().await.snapshot();
        let page_target = resolve_page_ordinal(&snapshot, arguments.page_ordinal)?;
        let page_id = page_target.id;
        let revision = snapshot.revision();
        snapshot.page(page_id)?;
        let original = self.project.original_for_page(&snapshot, page_id)?.clone();
        let inspection = self.inspect_snapshot(snapshot).await?;
        let page = inspection
            .project
            .pages
            .iter()
            .find(|page| page.id == page_id)
            .context("source-debugged page is missing from project inspection")?;
        let ordinals = page_reading_order_ordinals(page);
        let (bytes, labels) = render_page_source_debug_overlay(&original.bytes, page, &ordinals)?;
        let artifact = write_page_source_debug_artifact(
            &self.review_bundle_directory,
            revision,
            page_id,
            &bytes,
            labels,
        )?;
        *self.last_explicit_observation.lock() =
            Some(RevisionEvidence::SourcePageDebugInspection {
                revision,
                page_id,
                artifact_digest: artifact.blake3.clone(),
            });
        self.page_semantic_evidence.lock().record(
            PageEvidenceKind::SourceDebugArtifact,
            PageEvidenceArtifact {
                revision,
                page_id,
                blake3: artifact.blake3.clone(),
                element_crops: BTreeMap::new(),
            },
        );
        *self.current_semantic_evidence_page.lock() = Some(page_target);
        self.trace_records.lock().push(HostTraceRecord::new(
            "page_source_debug_view",
            serde_json::to_value(&artifact)?,
        ));
        let provenance = ToolImageProvenance::ContentHash {
            algorithm: "blake3",
            digest: artifact.blake3.clone(),
            media_type: artifact.media_type.to_owned(),
            byte_length: artifact.byte_length,
        };
        Ok(
            Invocation::read(serde_json::to_value(&artifact)?)?.with_image(
                format!(
                    "Original page source debug overlay {} ({page_id})",
                    page.label
                ),
                format!("data:image/png;base64,{}", STANDARD.encode(bytes)),
                provenance,
            ),
        )
    }

    async fn view_page_debug(&self, call: &ToolCall) -> Result<Invocation> {
        if !self.pipeline_completed() {
            bail!("the complete Koharu pipeline must finish before page debug review");
        }
        let arguments: ViewPageDebug = arguments(call)?;
        let snapshot = self.project.session().lock().await.snapshot();
        let page_target = resolve_page_ordinal(&snapshot, arguments.page_ordinal)?;
        let page_id = page_target.id;
        let revision = snapshot.revision();
        snapshot.page(page_id)?;
        let inspection = self.inspect_snapshot(snapshot.clone()).await?;
        let page = inspection
            .project
            .pages
            .iter()
            .find(|page| page.id == page_id)
            .context("debugged page is missing from project inspection")?;
        let normal_preview =
            rendered_preview(&self.renderer, self.rasterizer().await?, &snapshot, page_id).await?;
        let dossier = build_page_translation_dossier(
            revision,
            page,
            None,
            self.source_language.tag(),
            self.target_language.tag(),
            |_, current| current.clone(),
            RenderedPageReference {
                media_type: "image/webp",
                byte_length: normal_preview.len(),
                blake3: blake3::hash(&normal_preview).to_hex().to_string(),
            },
        );
        let ordinals = dossier
            .elements
            .iter()
            .map(|element| (element.ordinal, element.element_id))
            .collect::<Vec<_>>();
        let (bytes, labels) = render_page_debug_overlay(&normal_preview, page, &ordinals)?;
        let artifact = write_page_debug_artifact(
            &self.review_bundle_directory,
            revision,
            page_id,
            &bytes,
            labels,
        )?;
        *self.last_explicit_observation.lock() =
            Some(RevisionEvidence::RenderedPageDebugInspection {
                revision,
                page_id,
                artifact_digest: artifact.blake3.clone(),
            });
        self.page_semantic_evidence.lock().record(
            PageEvidenceKind::DebugArtifact,
            PageEvidenceArtifact {
                revision,
                page_id,
                blake3: artifact.blake3.clone(),
                element_crops: BTreeMap::new(),
            },
        );
        *self.current_semantic_evidence_page.lock() = Some(page_target);
        self.trace_records.lock().push(HostTraceRecord::new(
            "page_debug_view",
            serde_json::to_value(&artifact)?,
        ));
        let provenance = ToolImageProvenance::ContentHash {
            algorithm: "blake3",
            digest: artifact.blake3.clone(),
            media_type: artifact.media_type.to_owned(),
            byte_length: artifact.byte_length,
        };
        Ok(
            Invocation::read(serde_json::to_value(&artifact)?)?.with_image(
                format!("Page debug overlay {} ({page_id})", page.label),
                format!("data:image/png;base64,{}", STANDARD.encode(bytes)),
                provenance,
            ),
        )
    }

    async fn revise_page_translation(&self, call: &ToolCall) -> Result<Invocation> {
        self.validate_repair_workflow_state()?;
        let arguments: RevisePageTranslation = arguments(call)?;
        validate_page_translation_revision(&arguments, self.source_language, self.target_language)?;
        let requested_revision = Revision::new(arguments.evidence.scene_revision);
        let last_review = self.review_history.lock().last().cloned();

        let edits = arguments
            .edits
            .iter()
            .map(|edit| Ok((entity(&edit.element)?, edit)))
            .collect::<Result<Vec<_>>>()?;
        if let Some((element, _)) = edits.iter().find(|(element, _)| {
            self.decorative_sfx_decisions
                .lock()
                .get(element)
                .is_some_and(DecorativeSfxDecision::is_skipped)
        }) {
            bail!(
                "skipped difficult-SFX element {element} is retained source evidence and cannot receive source or target semantic edits"
            );
        }
        let active_failure = last_review
            .as_ref()
            .and_then(|review| review.deterministic_repair_plan.blocking_failures.first());
        validate_page_semantic_plan(active_failure, &edits)?;

        let mut session = self.project.session().lock().await;
        let snapshot = session.snapshot();
        let page_id = resolve_page_ordinal(&snapshot, arguments.page_ordinal)?.id;
        let current_revision = snapshot.revision();
        if requested_revision > current_revision {
            bail!(
                "page translation evidence is from future revision {}, current revision {}",
                requested_revision,
                current_revision
            );
        }
        snapshot.page(page_id)?;
        if !self
            .page_reviews
            .lock()
            .evidence_is_current(&self.page_semantic_evidence.lock(), page_id)
        {
            bail!("page translation evidence is stale for page {page_id}");
        }
        validate_page_translation_targets(
            &snapshot,
            page_id,
            &edits
                .iter()
                .map(|(element, _)| *element)
                .collect::<Vec<_>>(),
        )?;
        let evidence = page_translation_evidence(
            last_review.as_ref(),
            &self.page_semantic_evidence.lock(),
            current_revision,
            requested_revision,
            page_id,
            &arguments.evidence.source_evidence_dossier_blake3,
            &arguments.evidence.source_debug_artifact_blake3,
            &arguments.evidence.dossier_blake3,
            &arguments.evidence.debug_artifact_blake3,
        )?;
        let mut prepared = Vec::with_capacity(edits.len());
        let mut seen = std::collections::BTreeSet::new();
        for (element, edit) in &edits {
            if !seen.insert(*element) {
                bail!("page translation revision contains duplicate element {element}");
            }
            let (content_id, before) = repair_element_state(&snapshot, *element)?;
            prepared.push((*element, content_id, (*edit).clone(), before));
        }
        let source_language = LanguageTag::new(self.source_language.tag())?;
        let target_language = LanguageTag::new(self.target_language.tag())?;
        let patch = snapshot.patch(|editor| {
            for (_, content_id, edit, _) in &prepared {
                if let Some(source) = &edit.source {
                    editor.set(
                        *content_id,
                        &SourceText {
                            text: Authored::user(source.text.clone()),
                            language: Some(source_language.clone()),
                        },
                    )?;
                }
                if let Some(translation) = &edit.translation {
                    editor.set(
                        *content_id,
                        &Translation {
                            text: Authored::user(translation.text.clone()),
                            language: Some(target_language.clone()),
                        },
                    )?;
                }
            }
            Ok(())
        })?;
        let candidate = snapshot.preview([&patch])?;
        let mut after_states = Vec::with_capacity(prepared.len());
        for (element, _, _, before) in &prepared {
            let (_, after) = repair_element_state(&candidate, *element)?;
            let changed_fields = changed_repair_fields(before, &after);
            if changed_fields.is_empty() {
                bail!("page translation edit does not change element {element}");
            }
            if changed_fields
                .iter()
                .any(|field| !matches!(*field, "source_text" | "translation_text"))
            {
                bail!("page translation revision attempted a non-semantic mutation");
            }
            if let Some(failure) = active_failure {
                validate_planned_repair_fields(failure, &changed_fields)?;
            }
            after_states.push((after, changed_fields));
        }
        let revision_before = snapshot.revision();
        let revision_after = session.commit(patch).await?.snapshot.revision();
        drop(session);
        {
            let mut decisions = self.decorative_sfx_decisions.lock();
            for ((element, _, _, _), (after, changed_fields)) in prepared.iter().zip(&after_states)
            {
                if changed_fields.contains(&"source_text")
                    && let Some(decision) = decisions.get_mut(element)
                {
                    let source = after
                        .source
                        .as_ref()
                        .expect("validated source-text edit must retain committed source");
                    decision.source_ocr = SemanticText {
                        text: source.text.clone(),
                        language: source.language.clone(),
                    };
                    decision.decision_revision = revision_after;
                }
            }
        }
        self.record_page_mutation(page_id, revision_after);
        self.pending_text_safe_repairs.lock().clear();
        self.pending_text_layouts.lock().clear();
        self.pending_compact_translations.lock().clear();

        let mut corrections = self.corrections.lock();
        let mut records = Vec::with_capacity(prepared.len());
        for ((element, content_id, _, before), (after, changed_fields)) in
            prepared.into_iter().zip(after_states)
        {
            let previous_original = corrections
                .iter()
                .find(|record| record.element_id == element)
                .and_then(|record| record.original_ocr.clone());
            let record = CorrectionRecord {
                schema_version: CORRECTION_SCHEMA_VERSION,
                sequence: corrections.len() as u32 + 1,
                review_attempt: last_review.as_ref().map_or(0, |review| review.attempt),
                revision_before,
                revision_after,
                element_id: element,
                content_id,
                original_ocr: preserve_original_ocr(previous_original, before.source.clone()),
                evidence: evidence.clone(),
                changed_fields,
                before,
                after,
                reason: arguments.page_rationale.trim().to_owned(),
                agent_action: AgentAction {
                    actor: "codex_agent",
                    configured_model: self.configured_agent_model.clone(),
                    tool: "revise_page_translation",
                    tool_call_id: call.call_id.clone(),
                },
            };
            corrections.push(record.clone());
            records.push(record);
        }
        drop(corrections);
        *self.acceptance.lock() = None;
        *self.visual_review.lock() = None;
        self.trace_records.lock().push(HostTraceRecord::new(
            "page_translation_revision",
            json!({
                "schema_version": 3,
                "page_id": page_id,
                "revision_before": revision_before,
                "revision_after": revision_after,
                "evidence": evidence,
                "page_rationale": arguments.page_rationale.trim(),
                "corrections": records,
            }),
        ));
        Invocation::changed(json!({
            "page_id": page_id,
            "revision_before": revision_before,
            "revision_after": revision_after,
            "corrections": records,
            "next_action": "preview_text_layout for bubble fit, then review_pages",
        }))
    }

    async fn revise_element(&self, call: &ToolCall) -> Result<Invocation> {
        if !self.pipeline_completed() {
            bail!("the complete Koharu pipeline must finish before repair");
        }
        if let Some(diagnostic) = self.repair_stop.lock().clone() {
            bail!(
                "deterministic repair loop stopped: {}",
                serde_json::to_string(&diagnostic)?
            );
        }
        let arguments: ReviseElement = arguments(call)?;
        validate_revision_request(&arguments)?;
        let element = entity(&arguments.element)?;
        if self
            .decorative_sfx_decisions
            .lock()
            .get(&element)
            .is_some_and(DecorativeSfxDecision::is_skipped)
        {
            bail!(
                "skipped difficult-SFX element {element} has no target translation/render ownership and cannot be revised"
            );
        }
        let last_review = self
            .review_history
            .lock()
            .last()
            .cloned()
            .context("review_pages must evaluate the rendered revision before repair")?;
        let active_failure =
            planned_repair_failure(&last_review.deterministic_repair_plan, element)?;
        if active_failure.is_some_and(|failure| {
            failure.code == AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior
        }) {
            bail!(
                "text-safe-containment failures cannot use revise_element; call preview_increase_text_safe_padding with positive per-edge inset deltas, then commit_text_safe_layout_repair with the returned preview_id"
            );
        }
        if active_failure.is_none()
            && review_requires_layout_repair(&last_review, self.acceptance.lock().as_ref(), element)
            && arguments.typography.is_none()
            && arguments.layout.is_none()
        {
            bail!(
                "the rejected revision has a typography/layout failure; revise_element must change typography or layout instead of only rewriting text"
            );
        }
        let mut session = self.project.session().lock().await;
        let snapshot = session.snapshot();
        let element_page = page_containing(&snapshot, element)?;
        let evidence = repair_evidence(
            &last_review,
            self.last_explicit_observation.lock().as_ref(),
            snapshot.revision(),
            element_page,
        )?;
        let (content_id, before) = repair_element_state(&snapshot, element)?;
        let repaired_layout_geometry = arguments
            .layout
            .as_ref()
            .map(|layout| repair_layout_geometry(&snapshot, element, content_id, layout))
            .transpose()?
            .flatten();
        let source_language = LanguageTag::new(self.source_language.tag())?;
        let target_language = LanguageTag::new(self.target_language.tag())?;
        let patch = snapshot.patch(|edit| {
            if let Some(text) = &arguments.source_text {
                edit.set(
                    content_id,
                    &SourceText {
                        text: Authored::user(text.clone()),
                        language: Some(source_language.clone()),
                    },
                )?;
            }
            if let Some(text) = &arguments.translation_text {
                edit.set(
                    content_id,
                    &Translation {
                        text: Authored::user(text.clone()),
                        language: Some(target_language.clone()),
                    },
                )?;
            }
            if let Some(value) = &arguments.typography {
                edit.set(element, &value.to_scene(before.typography.as_ref()))?;
            }
            if let Some(layout) = &arguments.layout {
                if let Some(kind) = layout.kind {
                    edit.set(
                        element,
                        &TextLayout {
                            origin: koharu_scene::Origin::User,
                            kind: kind.into(),
                        },
                    )?;
                }
                if let Some(geometry) = &repaired_layout_geometry {
                    edit.set(element, geometry)?;
                }
            }
            Ok(())
        })?;
        let preview = snapshot.preview([&patch])?;
        let (_, after) = repair_element_state(&preview, element)?;
        let changed_fields = changed_repair_fields(&before, &after);
        if changed_fields.is_empty() {
            bail!("revision does not change the element");
        }
        if let Some(failure) = active_failure {
            validate_planned_repair_fields(failure, &changed_fields)?;
        }
        let revision_before = snapshot.revision();
        let revision_after = session.commit(patch).await?.snapshot.revision();
        drop(session);
        self.record_page_mutation(element_page, revision_after);
        self.pending_text_safe_repairs.lock().clear();
        self.pending_text_layouts.lock().clear();
        self.pending_compact_translations.lock().clear();

        let mut corrections = self.corrections.lock();
        let previous_original = corrections
            .iter()
            .find(|record| record.element_id == element)
            .and_then(|record| record.original_ocr.clone());
        let original_ocr = preserve_original_ocr(previous_original, before.source.clone());
        let record = CorrectionRecord {
            schema_version: CORRECTION_SCHEMA_VERSION,
            sequence: corrections.len() as u32 + 1,
            review_attempt: last_review.attempt,
            revision_before,
            revision_after,
            element_id: element,
            content_id,
            original_ocr,
            evidence,
            changed_fields,
            before,
            after,
            reason: arguments.reason.trim().to_owned(),
            agent_action: AgentAction {
                actor: "codex_agent",
                configured_model: self.configured_agent_model.clone(),
                tool: "revise_element",
                tool_call_id: call.call_id.clone(),
            },
        };
        corrections.push(record.clone());
        drop(corrections);
        *self.acceptance.lock() = None;
        *self.visual_review.lock() = None;
        self.trace_records.lock().push(HostTraceRecord::new(
            "agent_correction",
            serde_json::to_value(&record)?,
        ));
        Invocation::changed(json!({
            "correction": record,
            "review_required_for_revision": revision_after,
        }))
    }

    async fn preview_text_layout(&self, call: &ToolCall) -> Result<Invocation> {
        self.validate_repair_workflow_state()?;
        let arguments = preview_text_layout_arguments(call)?;
        let element = entity(&arguments.element)?;
        let last_review = self
            .review_history
            .lock()
            .last()
            .cloned()
            .context("review_pages must evaluate the rendered revision before repair")?;
        let active_failure =
            planned_repair_failure(&last_review.deterministic_repair_plan, element)?.cloned();
        validate_text_layout_plan(active_failure.as_ref(), &arguments.options)?;
        let session = self.project.session().lock().await;
        let snapshot = session.snapshot();
        let element_page = page_containing(&snapshot, element)?;
        let evidence = repair_evidence(
            &last_review,
            self.last_explicit_observation.lock().as_ref(),
            snapshot.revision(),
            element_page,
        )?;
        let (content_id, before_state) = repair_element_state(&snapshot, element)?;
        let current_translation = before_state
            .translation
            .as_ref()
            .context("preview_text_layout requires an existing translation")?;
        if current_translation.language.as_deref() != Some(self.target_language.tag()) {
            bail!("preview_text_layout requires the configured target-language translation");
        }
        let candidate_text = apply_line_break_policy(
            &current_translation.text,
            arguments.options.line_break_policy(),
            arguments.options.max_lines(),
        )?;
        let source_bound_fallback = arguments
            .options
            .source_bound_interjection_fallback()
            .map(|strategy| {
                source_bound_interjection_fallback_target(
                    &snapshot,
                    element,
                    content_id,
                    strategy,
                    self.free_dialogue_anchor_assessments.lock().values(),
                    &self.quality_thresholds,
                )
            })
            .transpose()?;
        if source_bound_fallback.is_some()
            && (self.ui_panel_anchors.lock().contains_key(&element)
                || self.free_dialogue_anchors.lock().contains_key(&element))
        {
            bail!("source-bound interjection fallback cannot replace a verified target anchor");
        }
        let candidate_typography = if source_bound_fallback.is_some() {
            Some(source_bound_interjection_typography(
                before_state.typography.as_ref(),
                self.quality_thresholds.min_rendered_font_size_px,
            )?)
        } else {
            candidate_text_layout_typography(before_state.typography.as_ref(), &arguments.options)?
        };
        let mut candidate_geometry = if let Some(geometry) = source_bound_fallback.clone() {
            Some(geometry)
        } else {
            arguments
                .options
                .safe_padding_increase_px()
                .map(|delta| {
                    increased_text_safe_padding_geometry(&snapshot, element, content_id, delta)
                })
                .transpose()?
                .map(|value| value.0)
        };
        let ui_panel_target = verified_ui_panel_layout_target(
            &snapshot,
            element,
            content_id,
            self.ui_panel_anchors.lock().get(&element),
        )?;
        let free_dialogue_target = verified_free_dialogue_layout_target(
            &snapshot,
            element,
            content_id,
            self.free_dialogue_anchors.lock().get(&element),
        )?;
        if ui_panel_target.is_some() && free_dialogue_target.is_some() {
            bail!("one text element cannot claim both UI-panel and free-dialogue anchors");
        }
        if source_bound_fallback.is_some()
            && (ui_panel_target.is_some() || free_dialogue_target.is_some())
        {
            bail!(
                "source-bound interjection fallback is available only when no adjacent target passes"
            );
        }
        let controlled_target = ui_panel_target.or(free_dialogue_target);
        let target_migration = controlled_target
            .map(|target| {
                let fit = snapshot.relation_from::<FitsTo>(element)?.context(
                    "verified target-anchor layout migration requires the source FitsTo relation",
                )?;
                let source_region = snapshot
                    .text_content(content_id)?
                    .source_region()?
                    .context("verified target-anchor layout migration requires a source region")?
                    .id();
                let add_inside = ui_panel_target.is_some()
                    && !snapshot
                        .relations_from_as::<Inside>(source_region)
                        .any(|relation| relation.value().target == target);
                let relation = if free_dialogue_target.is_some() {
                    if snapshot.analysis_region(target)?.region()?.kind != TextRegion::kind() {
                        bail!(
                            "verified free-dialogue target is not a text-safe text region for fits-to"
                        );
                    }
                    ControlledTargetRelation::FitsTo
                } else {
                    if snapshot.analysis_region(target)?.region()?.kind != PanelRegion::kind()
                    {
                        bail!(
                            "verified UI target is not the exact detector-backed panel region"
                        );
                    }
                    ControlledTargetRelation::FlowsIn
                };
                let (layout_geometry, layout_typography) = if free_dialogue_target.is_some() {
                    let bounds =
                        element_geometry(snapshot.analysis_region(target)?.geometry()?).bounds;
                    let baseline_reserve =
                        self.quality_thresholds.min_text_safe_padding_px * 0.5;
                    if !baseline_reserve.is_finite()
                        || baseline_reserve <= 0.0
                        || bounds.height <= baseline_reserve
                    {
                        bail!("verified free-dialogue target has no valid native layout frame");
                    }
                    let geometry = Geometry::rectangle(
                        bounds.x,
                        bounds.y + baseline_reserve,
                        bounds.width,
                        bounds.height - baseline_reserve,
                    );
                    let mut typography = candidate_typography
                        .as_ref()
                        .or(before_state.typography.as_ref())
                        .cloned()
                        .context("verified free-dialogue target requires typography intent")?;
                    typography.origin = koharu_scene::Origin::User;
                    typography.size = Some(
                        self.quality_thresholds.min_rendered_font_size_px as f32,
                    );
                    typography.auto_fit = false;
                    typography.writing_mode = Some(WritingMode::Horizontal);
                    (Some(geometry), Some(typography))
                } else {
                    (None, None)
                };
                Ok::<_, anyhow::Error>(ControlledTargetMigration {
                    target,
                    relation_to_remove: fit.id(),
                    source_region,
                    add_inside,
                    relation,
                    layout_geometry,
                    layout_typography,
                })
            })
            .transpose()?;
        let target_language = LanguageTag::new(self.target_language.tag())?;
        let build_patch = |geometry: &Option<Geometry>| {
            snapshot.patch(|edit| {
                if candidate_text != current_translation.text {
                    edit.set(
                        content_id,
                        &Translation {
                            text: Authored::user(candidate_text.clone()),
                            language: Some(target_language.clone()),
                        },
                    )?;
                }
                if let Some(typography) = &candidate_typography {
                    edit.set(element, typography)?;
                }
                if let Some(geometry) = geometry {
                    edit.set(element, geometry)?;
                }
                if let Some(migration) = target_migration.clone() {
                    apply_controlled_target_migration(edit, element, migration)?;
                }
                Ok(())
            })
        };
        let mut patch = build_patch(&candidate_geometry)?;
        let mut candidate = snapshot.preview([&patch])?;
        let current_acceptance = evaluate(&self.inspect_snapshot(snapshot.clone()).await?);
        let mut candidate_inspection = self.inspect_snapshot(candidate.clone()).await?;
        let mut candidate_acceptance = evaluate(&candidate_inspection);
        if source_bound_fallback.is_some() {
            let inspected = candidate_inspection
                .project
                .pages
                .iter()
                .flat_map(|page| &page.text_elements)
                .find(|candidate| candidate.id == element)
                .context("source-bound interjection final render is missing its element")?;
            let source_geometry = inspected
                .source_geometry
                .as_ref()
                .context("source-bound interjection final render is missing source geometry")?;
            let glyph_ink = inspected
                .final_scene
                .glyph_ink
                .as_ref()
                .context("source-bound interjection final render is missing glyph ink")?;
            let fitted = source_bound_interjection_render_fit(
                source_geometry,
                glyph_ink,
                self.quality_thresholds.min_text_safe_padding_px,
            )?;
            candidate_geometry = Some(fitted);
            patch = build_patch(&candidate_geometry)?;
            candidate = snapshot.preview([&patch])?;
            candidate_inspection = self.inspect_snapshot(candidate.clone()).await?;
            candidate_acceptance = evaluate(&candidate_inspection);
        }
        let (_, after_state) = repair_element_state(&candidate, element)?;
        let mut changed_fields = if source_bound_fallback.is_some() {
            validate_source_bound_interjection_changes(&before_state, &after_state)?;
            let mut changes = Vec::new();
            if !typography_intent_equal(
                before_state.typography.as_ref(),
                after_state.typography.as_ref(),
            ) {
                changes.extend([
                    "source_bound_interjection_vertical_writing_mode",
                    "source_bound_interjection_minimum_font",
                ]);
            }
            if !geometry_intent_equal(
                before_state.authored_layout_geometry.as_ref(),
                after_state.authored_layout_geometry.as_ref(),
            ) {
                changes.push("source_bound_interjection_evidence_inset");
            }
            changes
        } else {
            controlled_text_layout_changes(&before_state, &after_state)?
        };
        if ui_panel_target.is_some() {
            changed_fields.push("verified_ui_panel_target_anchor");
        }
        if free_dialogue_target.is_some() {
            changed_fields.push("source_raster_verified_adjacent_free_dialogue_anchor");
        }
        if changed_fields.is_empty() {
            bail!("text layout candidate does not change controlled layout intent");
        }
        let metrics_before = text_layout_metrics_for(&current_acceptance, element)?;
        let metrics_candidate = text_layout_metrics_for(&candidate_acceptance, element)?;
        if let Err(error) = validate_text_layout_candidate(
            &metrics_before,
            &metrics_candidate,
            arguments.options.max_lines(),
            &self.quality_thresholds,
        ) {
            drop(session);
            if active_failure
                .as_ref()
                .is_some_and(is_long_dialogue_target_anchor_failure)
            {
                self.expose_compact_translation_after_layout_failure(element, snapshot.revision())?;
                bail!(
                    "{error}; controlled layout is infeasible for the active long-dialogue target anchor, so the repair plan now exposes preview_compact_translation"
                );
            }
            return Err(error);
        }
        let preview_id = EntityId::new().to_string();
        let preview = TextLayoutPreview {
            preview_id: preview_id.clone(),
            operation: if ui_panel_target.is_some() {
                "controlled_verified_ui_panel_target_layout"
            } else if free_dialogue_target.is_some() {
                "controlled_source_raster_verified_free_dialogue_target_layout"
            } else if source_bound_fallback.is_some() {
                "controlled_source_bound_vertical_interjection_fallback"
            } else {
                "controlled_text_layout"
            },
            base_revision: snapshot.revision(),
            element_id: element,
            options: arguments.options.clone(),
            metrics_before,
            metrics_candidate,
            deterministic_constraints_satisfied: true,
            mutation_committed: false,
        };
        drop(session);
        store_pending_text_layout(
            &mut self.pending_text_layouts.lock(),
            preview_id.clone(),
            PreparedTextLayout {
                preview: preview.clone(),
                patch,
                content_id,
                evidence,
                before_state,
                after_state,
                reason: arguments.reason.trim().to_owned(),
                changed_fields,
            },
        );
        Invocation::read(json!({
            "preview": preview,
            "next_action": {
                "tool": "commit_text_layout",
                "arguments": { "preview_id": preview_id },
            },
        }))
    }

    fn expose_compact_translation_after_layout_failure(
        &self,
        element: EntityId,
        revision: Revision,
    ) -> Result<()> {
        let mut history = self.review_history.lock();
        let review = history
            .last_mut()
            .context("review_pages must establish an active layout failure")?;
        if review.scene_revision != revision {
            bail!("the active layout failure is stale for revision {revision}");
        }
        let failure = review
            .deterministic_repair_plan
            .blocking_failures
            .first_mut()
            .context("the active repair plan has no blocking failure")?;
        let required = failure.primary_render_element_id.or(failure.element_id);
        if required != Some(element) || !is_long_dialogue_target_anchor_failure(failure) {
            bail!("compact translation is not available for this layout failure");
        }
        failure.next_action = Some(compact_translation_next_action(element, revision));
        let updated = review.clone();
        drop(history);
        *self.visual_review.lock() = Some(updated);
        self.trace_records.lock().push(HostTraceRecord::new(
            "compact_translation_exposed",
            json!({
                "scene_revision": revision,
                "element_id": element,
                "reason": "controlled text layout preview could not satisfy deterministic constraints",
            }),
        ));
        Ok(())
    }

    fn expose_compact_translation_after_clearance_failure(
        &self,
        element: EntityId,
        revision: Revision,
    ) -> Result<()> {
        let mut history = self.review_history.lock();
        let review = history
            .last_mut()
            .context("review_pages must establish an active clearance failure")?;
        if review.scene_revision != revision {
            bail!("the active clearance failure is stale for revision {revision}");
        }
        let failure = review
            .deterministic_repair_plan
            .blocking_failures
            .first_mut()
            .context("the active repair plan has no blocking failure")?;
        let required = failure.primary_render_element_id.or(failure.element_id);
        if required != Some(element) || !is_long_dialogue_clearance_failure(failure) {
            bail!("compact translation is not available for this clearance failure");
        }
        failure.next_action = Some(compact_translation_next_action(element, revision));
        let updated = review.clone();
        drop(history);
        *self.visual_review.lock() = Some(updated);
        self.trace_records.lock().push(HostTraceRecord::new(
            "compact_translation_exposed",
            json!({
                "scene_revision": revision,
                "element_id": element,
                "reason": "text-safe inset reached clearance but could not preserve the unchanged minimum font gate",
            }),
        ));
        Ok(())
    }

    async fn commit_text_layout(&self, call: &ToolCall) -> Result<Invocation> {
        self.validate_repair_workflow_state()?;
        let arguments: CommitTextLayout = arguments(call)?;
        if arguments.preview_id.trim().is_empty() {
            bail!("preview_id cannot be empty");
        }
        let prepared = self
            .pending_text_layouts
            .lock()
            .get(&arguments.preview_id)
            .cloned()
            .with_context(|| {
                format!(
                    "unknown or expired text layout preview {}",
                    arguments.preview_id
                )
            })?;
        let last_review = self
            .review_history
            .lock()
            .last()
            .cloned()
            .context("review_pages must evaluate the rendered revision before repair")?;
        let active_failure = planned_repair_failure(
            &last_review.deterministic_repair_plan,
            prepared.preview.element_id,
        )?;
        validate_text_layout_plan(active_failure, &prepared.preview.options)?;

        let mut session = self.project.session().lock().await;
        let snapshot = session.snapshot();
        if snapshot.revision() != prepared.preview.base_revision {
            bail!(
                "text layout preview is stale: previewed revision {}, current revision {}",
                prepared.preview.base_revision,
                snapshot.revision()
            );
        }
        let candidate = snapshot.preview([&prepared.patch])?;
        let current_acceptance = evaluate(&self.inspect_snapshot(snapshot.clone()).await?);
        let candidate_acceptance = evaluate(&self.inspect_snapshot(candidate).await?);
        let metrics_before =
            text_layout_metrics_for(&current_acceptance, prepared.preview.element_id)?;
        let metrics_candidate =
            text_layout_metrics_for(&candidate_acceptance, prepared.preview.element_id)?;
        validate_text_layout_candidate(
            &metrics_before,
            &metrics_candidate,
            prepared.preview.options.max_lines(),
            &self.quality_thresholds,
        )?;
        let revision_before = snapshot.revision();
        let page_id = page_containing(&snapshot, prepared.preview.element_id)?;
        let revision_after = commit_text_layout_patch_if_safe(
            &mut session,
            prepared.patch.clone(),
            &metrics_before,
            &metrics_candidate,
            prepared.preview.options.max_lines(),
            &self.quality_thresholds,
        )
        .await?;
        drop(session);
        self.record_page_mutation(page_id, revision_after);
        self.pending_text_layouts.lock().clear();
        self.pending_text_safe_repairs.lock().clear();
        self.pending_compact_translations.lock().clear();

        let mut corrections = self.corrections.lock();
        let previous_original = corrections
            .iter()
            .find(|record| record.element_id == prepared.preview.element_id)
            .and_then(|record| record.original_ocr.clone());
        let record = CorrectionRecord {
            schema_version: CORRECTION_SCHEMA_VERSION,
            sequence: corrections.len() as u32 + 1,
            review_attempt: last_review.attempt,
            revision_before,
            revision_after,
            element_id: prepared.preview.element_id,
            content_id: prepared.content_id,
            original_ocr: preserve_original_ocr(
                previous_original,
                prepared.before_state.source.clone(),
            ),
            evidence: prepared.evidence,
            changed_fields: prepared.changed_fields,
            before: prepared.before_state,
            after: prepared.after_state,
            reason: prepared.reason,
            agent_action: AgentAction {
                actor: if call.call_id.starts_with(HOST_DETERMINISTIC_CALL_PREFIX) {
                    "koharu_host"
                } else {
                    "codex_agent"
                },
                configured_model: (!call.call_id.starts_with(HOST_DETERMINISTIC_CALL_PREFIX))
                    .then(|| self.configured_agent_model.clone())
                    .flatten(),
                tool: "commit_text_layout",
                tool_call_id: call.call_id.clone(),
            },
        };
        corrections.push(record.clone());
        drop(corrections);
        *self.acceptance.lock() = None;
        *self.visual_review.lock() = None;
        self.trace_records.lock().push(HostTraceRecord::new(
            "agent_correction",
            serde_json::to_value(&record)?,
        ));
        Invocation::changed(json!({
            "correction": record,
            "metrics_before": metrics_before,
            "metrics_after": metrics_candidate,
            "review_required_for_revision": revision_after,
        }))
    }

    async fn preview_compact_translation(&self, call: &ToolCall) -> Result<Invocation> {
        if !self.pipeline_completed() {
            bail!("the complete Koharu pipeline must finish before repair");
        }
        let arguments: PreviewCompactTranslation = arguments(call)?;
        validate_compact_translation_request(&arguments, self.target_language)?;
        let element = entity(&arguments.primary_render_element)?;
        let logical_group = entity(&arguments.logical_group)?;
        let requested_revision = Revision::new(arguments.evidence.scene_revision);

        let session = self.project.session().lock().await;
        let snapshot = session.snapshot();
        if snapshot.revision() != requested_revision {
            bail!(
                "compact translation evidence is stale: requested revision {}, current revision {}",
                requested_revision,
                snapshot.revision()
            );
        }
        let failure =
            self.active_compact_translation_failure(snapshot.revision(), element, logical_group)?;
        let page_id = page_containing(&snapshot, element)?;
        let evidence = compact_translation_evidence(
            &self.page_semantic_evidence.lock(),
            snapshot.revision(),
            page_id,
            &arguments.evidence,
            &failure.member_ordinal_ids,
        )?;
        let member_evidence = compact_member_evidence(
            &self.page_semantic_evidence.lock(),
            page_id,
            &failure.member_ordinal_ids,
            "preview_compact_translation",
        )?;
        let (content_id, before_state) = repair_element_state(&snapshot, element)?;
        let current_translation = before_state
            .translation
            .as_ref()
            .context("preview_compact_translation requires an existing translation")?;
        if current_translation.language.as_deref() != Some(self.target_language.tag()) {
            bail!(
                "preview_compact_translation requires the configured target-language translation"
            );
        }
        let candidate_text = arguments.candidate_translation.text.trim().to_owned();
        let (current_units, candidate_units) =
            validate_compact_translation_length(&current_translation.text, &candidate_text)?;
        let target_language = LanguageTag::new(self.target_language.tag())?;
        let patch = snapshot.patch(|edit| {
            edit.set(
                content_id,
                &Translation {
                    text: Authored::user(candidate_text.clone()),
                    language: Some(target_language),
                },
            )
        })?;
        let candidate = snapshot.preview([&patch])?;
        let (_, after_state) = repair_element_state(&candidate, element)?;
        validate_compact_translation_change(&before_state, &after_state)?;

        let current_inspection = self.inspect_snapshot(snapshot.clone()).await?;
        let candidate_inspection = self.inspect_snapshot(candidate).await?;
        validate_logical_group_unchanged(
            &current_inspection,
            &candidate_inspection,
            page_id,
            logical_group,
            element,
            &failure.member_ordinal_ids,
        )?;
        let current_acceptance = evaluate(&current_inspection);
        let candidate_acceptance = evaluate(&candidate_inspection);
        let metrics_before = text_layout_metrics_for(&current_acceptance, element)?;
        let metrics_candidate = text_layout_metrics_for(&candidate_acceptance, element)?;
        validate_compact_translation_candidate(
            &metrics_before,
            &metrics_candidate,
            &self.quality_thresholds,
        )?;

        let preview_id = EntityId::new().to_string();
        let preview = CompactTranslationPreview {
            preview_id: preview_id.clone(),
            operation: "layout_constrained_translation_compaction",
            base_revision: snapshot.revision(),
            logical_group_id: logical_group,
            primary_render_element_id: element,
            member_ordinal_ids: failure.member_ordinal_ids.clone(),
            current_visible_grapheme_units: current_units,
            candidate_visible_grapheme_units: candidate_units,
            metrics_before,
            metrics_candidate,
            deterministic_constraints_satisfied: true,
            mutation_committed: false,
        };
        drop(session);
        self.pending_compact_translations.lock().clear();
        self.pending_compact_translations.lock().insert(
            preview_id.clone(),
            PreparedCompactTranslation {
                preview: preview.clone(),
                patch,
                content_id,
                evidence,
                before_state,
                after_state,
                reason: arguments.rationale.trim().to_owned(),
                member_ordinal_ids: failure.member_ordinal_ids,
            },
        );
        Invocation::read(json!({
            "preview": preview,
            "required_source_member_evidence": member_evidence,
            "next_action": {
                "tool": "commit_compact_translation",
                "arguments": { "preview_id": preview_id },
            },
        }))
    }

    async fn commit_compact_translation(&self, call: &ToolCall) -> Result<Invocation> {
        if !self.pipeline_completed() {
            bail!("the complete Koharu pipeline must finish before repair");
        }
        let arguments: CommitCompactTranslation = arguments(call)?;
        if arguments.preview_id.trim().is_empty() {
            bail!("preview_id cannot be empty");
        }
        let prepared = self
            .pending_compact_translations
            .lock()
            .get(&arguments.preview_id)
            .cloned()
            .with_context(|| {
                format!(
                    "unknown or expired compact translation preview {}",
                    arguments.preview_id
                )
            })?;

        let mut session = self.project.session().lock().await;
        let snapshot = session.snapshot();
        if snapshot.revision() != prepared.preview.base_revision {
            bail!(
                "compact translation preview is stale: previewed revision {}, current revision {}",
                prepared.preview.base_revision,
                snapshot.revision()
            );
        }
        let failure = self.active_compact_translation_failure(
            snapshot.revision(),
            prepared.preview.primary_render_element_id,
            prepared.preview.logical_group_id,
        )?;
        if failure.member_ordinal_ids != prepared.member_ordinal_ids {
            bail!("compact translation preview is stale because logical group membership changed");
        }
        validate_prepared_compact_evidence(
            &self.page_semantic_evidence.lock(),
            snapshot.revision(),
            &prepared.evidence,
            &failure.member_ordinal_ids,
        )?;
        let candidate = snapshot.preview([&prepared.patch])?;
        let page_id = page_containing(&snapshot, prepared.preview.primary_render_element_id)?;
        let current_inspection = self.inspect_snapshot(snapshot.clone()).await?;
        let candidate_inspection = self.inspect_snapshot(candidate).await?;
        validate_logical_group_unchanged(
            &current_inspection,
            &candidate_inspection,
            page_id,
            prepared.preview.logical_group_id,
            prepared.preview.primary_render_element_id,
            &failure.member_ordinal_ids,
        )?;
        let current_acceptance = evaluate(&current_inspection);
        let candidate_acceptance = evaluate(&candidate_inspection);
        let metrics_before = text_layout_metrics_for(
            &current_acceptance,
            prepared.preview.primary_render_element_id,
        )?;
        let metrics_candidate = text_layout_metrics_for(
            &candidate_acceptance,
            prepared.preview.primary_render_element_id,
        )?;
        let revision_before = snapshot.revision();
        let revision_after = commit_compact_translation_patch_if_safe(
            &mut session,
            prepared.patch.clone(),
            prepared.preview.base_revision,
            &metrics_before,
            &metrics_candidate,
            &self.quality_thresholds,
        )
        .await?;
        drop(session);
        self.record_page_mutation(page_id, revision_after);

        self.pending_compact_translations.lock().clear();
        self.pending_text_layouts.lock().clear();
        self.pending_text_safe_repairs.lock().clear();
        *self.repair_stop.lock() = None;

        let mut corrections = self.corrections.lock();
        let previous_original = corrections
            .iter()
            .find(|record| record.element_id == prepared.preview.primary_render_element_id)
            .and_then(|record| record.original_ocr.clone());
        let record = CorrectionRecord {
            schema_version: CORRECTION_SCHEMA_VERSION,
            sequence: corrections.len() as u32 + 1,
            review_attempt: self
                .review_history
                .lock()
                .last()
                .map_or(0, |review| review.attempt),
            revision_before,
            revision_after,
            element_id: prepared.preview.primary_render_element_id,
            content_id: prepared.content_id,
            original_ocr: preserve_original_ocr(
                previous_original,
                prepared.before_state.source.clone(),
            ),
            evidence: prepared.evidence,
            changed_fields: vec!["translation_text"],
            before: prepared.before_state,
            after: prepared.after_state,
            reason: prepared.reason,
            agent_action: AgentAction {
                actor: "codex_agent",
                configured_model: self.configured_agent_model.clone(),
                tool: "commit_compact_translation",
                tool_call_id: call.call_id.clone(),
            },
        };
        corrections.push(record.clone());
        drop(corrections);
        *self.acceptance.lock() = None;
        *self.visual_review.lock() = None;
        self.trace_records.lock().push(HostTraceRecord::new(
            "compact_translation_correction",
            serde_json::to_value(&record)?,
        ));
        Invocation::changed(json!({
            "correction": record,
            "logical_group_id": prepared.preview.logical_group_id,
            "member_ordinal_ids": prepared.member_ordinal_ids,
            "metrics_before": metrics_before,
            "metrics_after": metrics_candidate,
            "review_required_for_revision": revision_after,
            "next_action": "refresh all four evidence parts, run review_pages, then explicitly review compacted meaning and Korean naturalness against every original group-member crop",
        }))
    }

    async fn revise_compact_translation(&self, call: &ToolCall) -> Result<Invocation> {
        let preview = self.preview_compact_translation(call).await?;
        let preview_id = preview
            .value
            .pointer("/preview/preview_id")
            .and_then(Value::as_str)
            .context("compact translation preview did not issue an exact preview ID")?
            .to_owned();
        let commit_call = ToolCall {
            call_id: format!("{}-host-commit", call.call_id),
            name: "commit_compact_translation".to_owned(),
            arguments: json!({ "preview_id": preview_id }).to_string(),
        };
        let commit = self.commit_compact_translation(&commit_call).await?;
        let review = self.record_review_and_execute_host_repairs().await?;
        Invocation::changed(json!({
            "semantic_revision_preview": preview.value,
            "semantic_revision_commit": commit.value,
            "visual_review": review,
            "host_repair_stop_diagnostic": self.repair_stop.lock().clone(),
            "next_action": if self.repair_stop.lock().is_some() {
                "inspect_project for the terminal host repair diagnostic"
            } else {
                "follow the current host-reviewed workflow surface"
            },
        }))
    }

    fn active_compact_translation_failure(
        &self,
        current_revision: Revision,
        requested_element: EntityId,
        requested_group: EntityId,
    ) -> Result<DeterministicRepairFailure> {
        let last_review = self
            .review_history
            .lock()
            .last()
            .cloned()
            .context("review_pages must evaluate the rendered revision before repair")?;
        let failure = if let Some(diagnostic) = self.repair_stop.lock().clone() {
            if diagnostic.reviewed_revision != current_revision {
                bail!(
                    "the stopped repair diagnostic is stale for current revision {current_revision}"
                );
            }
            diagnostic
                .first_unresolved_failure
                .context("the stopped repair diagnostic has no active failure")?
        } else {
            if last_review.scene_revision != current_revision {
                bail!(
                    "review_pages must evaluate current revision {current_revision} before repair"
                );
            }
            planned_repair_failure(&last_review.deterministic_repair_plan, requested_element)?
                .cloned()
                .context("preview_compact_translation requires an active deterministic failure")?
        };
        validate_compact_translation_plan(&failure, requested_element, requested_group)?;
        Ok(failure)
    }

    async fn preview_text_safe_layout_repair(&self, call: &ToolCall) -> Result<Invocation> {
        self.validate_repair_workflow_state()?;
        let arguments: PreviewTextSafeLayoutRepair = arguments(call)?;
        validate_text_safe_inset_delta(arguments.inset_delta_px)?;
        if arguments.reason.trim().is_empty() {
            bail!("text-safe layout repair reason cannot be empty");
        }
        let element = entity(&arguments.element)?;
        let last_review = self
            .review_history
            .lock()
            .last()
            .cloned()
            .context("review_pages must evaluate the rendered revision before repair")?;
        let failure = planned_repair_failure(&last_review.deterministic_repair_plan, element)?
            .context(
                "preview_increase_text_safe_padding requires an active deterministic failure",
            )?;
        if failure.code != AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior {
            bail!(
                "the active deterministic failure is {}; use its repair-plan next_action",
                rejection_code_name(failure.code)
            );
        }
        let session = self.project.session().lock().await;
        let snapshot = session.snapshot();
        if snapshot.revision() != failure.required_evidence_revision {
            bail!(
                "text-safe repair requires reviewed revision {}, but current revision is {}",
                failure.required_evidence_revision,
                snapshot.revision()
            );
        }
        let element_page = page_containing(&snapshot, element)?;
        let evidence = repair_evidence(
            &last_review,
            self.last_explicit_observation.lock().as_ref(),
            snapshot.revision(),
            element_page,
        )?;
        let (content_id, before_state) = repair_element_state(&snapshot, element)?;
        let (geometry, layout_bounds_before, layout_bounds_candidate) =
            increased_text_safe_padding_geometry(
                &snapshot,
                element,
                content_id,
                arguments.inset_delta_px,
            )?;
        let patch = snapshot.patch(|edit| edit.set(element, &geometry))?;
        let candidate = snapshot.preview([&patch])?;
        let (_, after_state) = repair_element_state(&candidate, element)?;
        let current_acceptance = evaluate(&self.inspect_snapshot(snapshot.clone()).await?);
        let candidate_acceptance = evaluate(&self.inspect_snapshot(candidate).await?);
        let clearance_before = text_safe_containment_for(&current_acceptance, element)?.clone();
        let clearance_candidate =
            text_safe_containment_for(&candidate_acceptance, element)?.clone();
        let metrics_before = text_layout_metrics_for(&current_acceptance, element)?;
        let metrics_candidate = text_layout_metrics_for(&candidate_acceptance, element)?;
        let deterministic_diagnostics =
            deterministic_diagnostics_for(&candidate_acceptance, element);
        let validation = validate_text_safe_layout_candidate(
            &metrics_before,
            &metrics_candidate,
            &self.quality_thresholds,
        );

        let preview_id = EntityId::new().to_string();
        let preview = TextSafeLayoutRepairPreview {
            preview_id: preview_id.clone(),
            operation: "increase_text_safe_padding",
            base_revision: snapshot.revision(),
            element_id: element,
            inset_delta_px: arguments.inset_delta_px,
            layout_bounds_before,
            layout_bounds_candidate,
            clearance_before,
            clearance_candidate,
            metrics_before,
            metrics_candidate,
            deterministic_diagnostics,
            deterministic_constraints_satisfied: validation.is_ok(),
            committable: validation.is_ok(),
            mutation_committed: false,
        };
        drop(session);
        if let Err(error) = validation {
            self.pending_text_safe_repairs.lock().clear();
            let compact_translation_exposed = preview
                .metrics_candidate
                .deterministic_rejection_codes
                .iter()
                .any(|code| code == "rendered_font_size_below_minimum")
                && is_long_dialogue_clearance_failure(failure)
                && self
                    .expose_compact_translation_after_clearance_failure(
                        element,
                        preview.base_revision,
                    )
                    .is_ok();
            return Invocation::read(json!({
                "preview": preview,
                "rejected": true,
                "rejection": {
                    "code": "candidate_failed_global_deterministic_gate",
                    "message": error.to_string(),
                    "diagnostics": preview.deterministic_diagnostics.clone(),
                },
                "next_action": compact_translation_exposed.then(|| json!({
                    "tool": "preview_compact_translation",
                    "element": element,
                    "logical_group": failure.logical_group_id,
                    "required_evidence_revision": preview.base_revision,
                })),
            }));
        }
        self.pending_text_safe_repairs.lock().clear();
        self.pending_text_safe_repairs.lock().insert(
            preview_id.clone(),
            PreparedTextSafeLayoutRepair {
                preview: preview.clone(),
                patch,
                content_id,
                evidence,
                before_state,
                after_state,
                reason: arguments.reason.trim().to_owned(),
            },
        );
        Invocation::read(json!({
            "preview": preview,
            "next_action": {
                "tool": "commit_text_safe_layout_repair",
                "arguments": { "preview_id": preview_id },
            },
        }))
    }

    async fn commit_text_safe_layout_repair(&self, call: &ToolCall) -> Result<Invocation> {
        self.validate_repair_workflow_state()?;
        let arguments: CommitTextSafeLayoutRepair = arguments(call)?;
        if arguments.preview_id.trim().is_empty() {
            bail!("preview_id cannot be empty");
        }
        let prepared = self
            .pending_text_safe_repairs
            .lock()
            .get(&arguments.preview_id)
            .cloned()
            .with_context(|| {
                format!(
                    "unknown or expired text-safe repair preview {}",
                    arguments.preview_id
                )
            })?;
        let last_review = self
            .review_history
            .lock()
            .last()
            .cloned()
            .context("review_pages must evaluate the rendered revision before repair")?;
        let failure = planned_repair_failure(
            &last_review.deterministic_repair_plan,
            prepared.preview.element_id,
        )?
        .context("the prepared text-safe repair no longer has an active deterministic failure")?;
        if failure.code != AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior {
            bail!("the prepared preview no longer matches the active deterministic failure");
        }

        let mut session = self.project.session().lock().await;
        let snapshot = session.snapshot();
        if snapshot.revision() != prepared.preview.base_revision
            || snapshot.revision() != failure.required_evidence_revision
        {
            bail!(
                "text-safe repair preview is stale: previewed revision {}, current revision {}",
                prepared.preview.base_revision,
                snapshot.revision()
            );
        }
        let candidate = snapshot.preview([&prepared.patch])?;
        let current_acceptance = evaluate(&self.inspect_snapshot(snapshot.clone()).await?);
        let candidate_acceptance = evaluate(&self.inspect_snapshot(candidate).await?);
        let metrics_before =
            text_layout_metrics_for(&current_acceptance, prepared.preview.element_id)?;
        let metrics_candidate =
            text_layout_metrics_for(&candidate_acceptance, prepared.preview.element_id)?;
        validate_text_safe_layout_candidate(
            &metrics_before,
            &metrics_candidate,
            &self.quality_thresholds,
        )?;
        let clearance_before = metrics_before
            .text_safe_clearance
            .as_ref()
            .context("current text-safe clearance is unavailable during commit")?;
        let clearance_candidate = metrics_candidate
            .text_safe_clearance
            .as_ref()
            .context("candidate text-safe clearance is unavailable during commit")?;

        let revision_before = snapshot.revision();
        let page_id = page_containing(&snapshot, prepared.preview.element_id)?;
        let revision_after = commit_text_safe_patch_if_globally_safe(
            &mut session,
            prepared.patch.clone(),
            prepared.preview.base_revision,
            &metrics_before,
            &metrics_candidate,
            &self.quality_thresholds,
        )
        .await?;
        drop(session);
        self.record_page_mutation(page_id, revision_after);
        self.pending_text_safe_repairs.lock().clear();
        self.pending_text_layouts.lock().clear();
        self.pending_compact_translations.lock().clear();

        let mut corrections = self.corrections.lock();
        let previous_original = corrections
            .iter()
            .find(|record| record.element_id == prepared.preview.element_id)
            .and_then(|record| record.original_ocr.clone());
        let original_ocr =
            preserve_original_ocr(previous_original, prepared.before_state.source.clone());
        let record = CorrectionRecord {
            schema_version: CORRECTION_SCHEMA_VERSION,
            sequence: corrections.len() as u32 + 1,
            review_attempt: last_review.attempt,
            revision_before,
            revision_after,
            element_id: prepared.preview.element_id,
            content_id: prepared.content_id,
            original_ocr,
            evidence: prepared.evidence,
            changed_fields: vec!["layout_geometry"],
            before: prepared.before_state,
            after: prepared.after_state,
            reason: prepared.reason,
            agent_action: AgentAction {
                actor: if call.call_id.starts_with(HOST_DETERMINISTIC_CALL_PREFIX) {
                    "koharu_host"
                } else {
                    "codex_agent"
                },
                configured_model: (!call.call_id.starts_with(HOST_DETERMINISTIC_CALL_PREFIX))
                    .then(|| self.configured_agent_model.clone())
                    .flatten(),
                tool: "commit_text_safe_layout_repair",
                tool_call_id: call.call_id.clone(),
            },
        };
        corrections.push(record.clone());
        drop(corrections);
        *self.acceptance.lock() = None;
        *self.visual_review.lock() = None;
        self.trace_records.lock().push(HostTraceRecord::new(
            "agent_correction",
            serde_json::to_value(&record)?,
        ));
        Invocation::changed(json!({
            "correction": record,
            "clearance_before": clearance_before,
            "clearance_after": clearance_candidate,
            "review_required_for_revision": revision_after,
        }))
    }

    fn validate_repair_workflow_state(&self) -> Result<()> {
        if !self.pipeline_completed() {
            bail!("the complete Koharu pipeline must finish before repair");
        }
        if let Some(diagnostic) = self.repair_stop.lock().clone() {
            bail!(
                "deterministic repair loop stopped: {}",
                serde_json::to_string(&diagnostic)?
            );
        }
        Ok(())
    }

    async fn run_source_raster_panel_detection(&self) -> Result<Vec<Value>> {
        let snapshot = self.project.session().lock().await.snapshot();
        let generation = Generation {
            producer: ProducerId::new(RASTER_PANEL_DETECTOR)?,
            model: Some(RASTER_PANEL_DETECTOR_VERSION.to_owned()),
            confidence: None,
        };
        let mut edit = snapshot.edit_as(generation.clone());
        let mut accepted = Vec::<(EntityId, RasterPanelEvidence)>::new();
        let mut page_reports = Vec::new();

        for page in snapshot.pages() {
            let page_id = page.id();
            let original = self.project.original_for_page(&snapshot, page_id)?;
            let raster = image::load_from_memory(&original.bytes)
                .with_context(|| {
                    format!("failed to decode original source page for panel detection {page_id}")
                })?
                .to_luma8();
            let mut assessments = Vec::<RasterPanelAssessment>::new();
            for entity in snapshot.descendants(page_id)? {
                let Some(region) = entity.component::<Region>()? else {
                    continue;
                };
                if region.kind != TextRegion::kind()
                    || entity.component::<DetectionAnalysis>()?.is_none()
                {
                    continue;
                }
                let Some(geometry) = entity.component::<Geometry>()? else {
                    continue;
                };
                let assessment = assess_source_raster_panel(
                    &raster,
                    entity.id(),
                    element_geometry(geometry).bounds,
                );
                if let Some(evidence) = assessment.accepted.clone() {
                    let bounds = evidence.panel_bbox;
                    let panel = edit.add_entity(page_id, At::End)?;
                    edit.set(
                        panel,
                        &Geometry::rectangle(bounds.x, bounds.y, bounds.width, bounds.height),
                    )?;
                    edit.set(
                        panel,
                        &Region {
                            origin: koharu_scene::Origin::Generated(generation.clone()),
                            kind: PanelRegion::kind(),
                            label: Some("source-raster-closed-ui-panel".to_owned()),
                        },
                    )?;
                    edit.set(
                        panel,
                        &DetectionAnalysis {
                            origin: koharu_scene::Origin::Generated(generation.clone()),
                            labels: vec![DetectionLabel {
                                kind: PanelRegion::kind(),
                                confidence: evidence.confidence,
                            }],
                        },
                    )?;
                    edit.relate::<Inside>(entity.id(), panel)?;
                    accepted.push((panel, evidence));
                }
                assessments.push(assessment);
            }
            page_reports.push(json!({
                "page_id": page_id,
                "detector": {
                    "producer": RASTER_PANEL_DETECTOR,
                    "version": RASTER_PANEL_DETECTOR_VERSION,
                    "input": "original_source_pixels",
                },
                "assessments": assessments,
            }));
        }

        if !accepted.is_empty() {
            let patch = edit
                .finish()?
                .with_label("Detect closed UI panels from original source pixels");
            self.project.session().lock().await.commit(patch).await?;
            self.raster_panel_evidence.lock().extend(accepted);
        }
        Ok(page_reports)
    }

    async fn bind_deterministic_ui_panel_anchors(&self) -> Result<Vec<UiPanelAnchorDecision>> {
        let snapshot = self.project.session().lock().await.snapshot();
        let evidence_revision = snapshot.revision();
        let decision_revision = evidence_revision
            .next()
            .context("revision overflow while preparing deterministic UI-panel anchors")?;
        let inspection = self.inspect_snapshot(snapshot.clone()).await?;
        let mut prepared = Vec::new();

        for page in &inspection.project.pages {
            let original = self.project.original_for_page(&snapshot, page.id)?;
            let mut ocr_confidences = BTreeMap::new();
            for region in page
                .text_elements
                .iter()
                .filter_map(|element| element.source_region_id)
            {
                let confidence = snapshot
                    .component::<OcrAnalysis>(region)?
                    .and_then(|analysis| analysis.confidence);
                ocr_confidences.insert(region, confidence);
            }
            let dossier = build_source_evidence_dossier(
                &self.review_bundle_directory,
                evidence_revision,
                page,
                &original.bytes,
                &original.media_type,
                |_element, current| current.clone(),
                |region| region.and_then(|id| ocr_confidences.get(&id).copied().flatten()),
            )?;
            let source_evidence = dossier
                .elements
                .iter()
                .map(|element| {
                    (
                        element.element_id,
                        SourceElementEvidence {
                            ordinal: element.ordinal,
                            source_debug_label: element.source_debug_label.clone(),
                            crop_blake3: element.original_crop.blake3.clone(),
                        },
                    )
                })
                .collect();
            prepared.extend(prepare_deterministic_ui_panel_bindings(
                page,
                &source_evidence,
                evidence_revision,
                decision_revision,
            )?);
        }

        if prepared.is_empty() {
            return Ok(Vec::new());
        }
        let mut edit = snapshot.edit();
        for decision in &prepared {
            edit.set(
                decision.content_id,
                &TextRole {
                    origin: koharu_scene::Origin::Generated(Generation {
                        producer: ProducerId::new(RASTER_PANEL_DETECTOR)?,
                        model: Some(RASTER_PANEL_DETECTOR_VERSION.to_owned()),
                        confidence: Some(decision.confidence),
                    }),
                    role: UI_TEXT_ROLE.to_owned(),
                },
            )?;
        }
        let patch = edit
            .finish()?
            .with_label("Bind exact source-raster UI panel text anchors");
        let committed = self.project.session().lock().await.commit(patch).await?;
        ensure!(
            committed.snapshot.revision() == decision_revision,
            "deterministic UI-panel binding committed an unexpected revision"
        );
        {
            let mut recorded = self.ui_panel_anchors.lock();
            for decision in &prepared {
                recorded.insert(decision.element_id, decision.clone());
            }
        }
        self.trace_records.lock().push(HostTraceRecord::new(
            "ui_panel_anchor_verification",
            json!({
                "schema_version": 3,
                "binding_owner": "host_deterministic_source_analysis",
                "scene_revision_before": evidence_revision,
                "scene_revision_after": decision_revision,
                "decisions": prepared,
            }),
        ));
        Ok(prepared)
    }

    async fn run_source_raster_free_dialogue_anchor_detection(&self) -> Result<Vec<Value>> {
        let snapshot = self.project.session().lock().await.snapshot();
        let evidence_revision = snapshot.revision();
        let decision_revision = evidence_revision
            .next()
            .context("revision overflow while preparing free-dialogue anchors")?;
        let generation = Generation {
            producer: ProducerId::new(RASTER_FREE_DIALOGUE_DETECTOR)?,
            model: Some(RASTER_FREE_DIALOGUE_DETECTOR_VERSION.to_owned()),
            confidence: None,
        };
        let anchor_detection_kind = RegionKind::new(ADJACENT_FREE_DIALOGUE_DETECTION_KIND)?;
        let mut edit = snapshot.edit_as(generation.clone());
        let mut decisions = BTreeMap::new();
        let mut assessments_by_source = BTreeMap::new();
        let mut page_reports = Vec::new();

        for page in snapshot.pages() {
            let page_id = page.id();
            let original = self.project.original_for_page(&snapshot, page_id)?;
            let raster = image::load_from_memory(&original.bytes)
                .with_context(|| {
                    format!(
                        "failed to decode original source page for free-dialogue anchor detection {page_id}"
                    )
                })?
                .to_luma8();
            let Some(group) = page.text_group()? else {
                continue;
            };
            let layers = group.text_layers()?.collect::<Vec<_>>();
            let source_bounds = layers
                .iter()
                .filter_map(|layer| {
                    let content = layer.content().ok()?;
                    let source = content.source_region().ok()??;
                    source
                        .geometry()
                        .ok()
                        .map(|geometry| (source.id(), element_geometry(geometry).bounds))
                })
                .collect::<Vec<_>>();
            let mut page_assessments = Vec::new();
            for layer in layers {
                let element_id = layer.id();
                let content = layer.content()?;
                let content_id = content.id();
                let Some(source_region) = content.source_region()? else {
                    continue;
                };
                if source_region.region()?.kind != TextRegion::kind()
                    || source_region.detection()?.is_none()
                {
                    continue;
                }
                let source_region_id = source_region.id();
                let bounds = element_geometry(source_region.geometry()?).bounds;
                let source = content.source()?;
                let translation = content.translation()?;
                let role = content.role()?.map(|role| role.role).unwrap_or_default();
                let source_mode = snapshot
                    .component::<OcrAnalysis>(source_region_id)?
                    .and_then(|analysis| match analysis.direction {
                        TextDirection::Vertical => Some(WritingMode::Vertical),
                        TextDirection::Horizontal => Some(WritingMode::Horizontal),
                        TextDirection::Auto => None,
                    });
                let typography = layer.typography()?;
                let target_mode = typography
                    .as_ref()
                    .and_then(|typography| typography.writing_mode);
                let target_ink_luma = typography
                    .as_ref()
                    .and_then(|typography| typography.color)
                    .map_or(0, |[red, green, blue, _]| {
                        (0.2126 * f64::from(red)
                            + 0.7152 * f64::from(green)
                            + 0.0722 * f64::from(blue))
                        .round() as u8
                    });
                let other_source_bounds = source_bounds
                    .iter()
                    .filter_map(|(id, bounds)| (*id != source_region_id).then_some(*bounds))
                    .collect::<Vec<_>>();
                let has_container_relation = layer.balloon_target()?.is_some()
                    || self.ui_panel_anchors.lock().contains_key(&element_id);
                let assessment = assess_free_dialogue_anchor(
                    &raster,
                    FreeDialogueAnchorInput {
                        source_region_id,
                        source_bounds: bounds,
                        source_text: source.as_ref().map_or("", |text| text.text.value.as_str()),
                        target_text: translation
                            .as_ref()
                            .map_or("", |text| text.text.value.as_str()),
                        source_scene_role: &role,
                        required: self
                            .decorative_sfx_decisions
                            .lock()
                            .get(&element_id)
                            .is_none_or(DecorativeSfxDecision::requires_translation),
                        has_finite_verified_container_relation: has_container_relation,
                        source_language: source
                            .as_ref()
                            .and_then(|text| text.language.as_ref())
                            .map(ToString::to_string)
                            .as_deref(),
                        target_language: translation
                            .as_ref()
                            .and_then(|text| text.language.as_ref())
                            .map(ToString::to_string)
                            .as_deref(),
                        source_writing_mode: source_mode,
                        target_writing_mode: target_mode,
                        target_ink_luma,
                        other_source_bounds: &other_source_bounds,
                    },
                );
                if let (Some(source), Some(translation), Some(selected)) =
                    (source, translation, assessment.selected_candidate.clone())
                {
                    let target_region_id = edit.add_entity(page_id, At::End)?;
                    edit.set(
                        target_region_id,
                        &Geometry::rectangle(
                            selected.bounds.x,
                            selected.bounds.y,
                            selected.bounds.width,
                            selected.bounds.height,
                        ),
                    )?;
                    edit.set(
                        target_region_id,
                        &Region {
                            origin: koharu_scene::Origin::Generated(generation.clone()),
                            kind: TextRegion::kind(),
                            label: Some("source-raster-adjacent-free-dialogue-anchor".to_owned()),
                        },
                    )?;
                    edit.set(
                        target_region_id,
                        &DetectionAnalysis {
                            origin: koharu_scene::Origin::Generated(generation.clone()),
                            labels: vec![DetectionLabel {
                                kind: anchor_detection_kind.clone(),
                                confidence: selected.association.confidence as f32,
                            }],
                        },
                    )?;
                    decisions.insert(
                        element_id,
                        FreeDialogueAnchorDecision {
                            schema_version: FREE_DIALOGUE_ANCHOR_SCHEMA_VERSION,
                            decision: "source_raster_verified_adjacent_free_dialogue_anchor",
                            evidence_revision,
                            decision_revision,
                            page_id,
                            element_id,
                            content_id,
                            source_region_id,
                            target_region_id,
                            source_ocr: SemanticText {
                                text: source.text.value,
                                language: source.language.map(|language| language.to_string()),
                            },
                            target_translation: SemanticText {
                                text: translation.text.value,
                                language: translation.language.map(|language| language.to_string()),
                            },
                            source_bounds: bounds,
                            candidate_bounds: selected.bounds,
                            distance_px: selected.distance_px,
                            direction: selected.direction,
                            pixel_analysis: selected.pixels.clone(),
                            contrast_background_evidence: selected.contrast.clone(),
                            room_evidence: selected.room.clone(),
                            association: selected.association.clone(),
                            writing_modes: assessment.writing_mode_gate.clone(),
                            role_gate: assessment.role_gate.clone(),
                            deterministic_score: selected.score,
                            deterministic_score_threshold: selected.score_threshold,
                            association_confidence: selected.association.confidence,
                            association_reason: selected.association.reason.clone(),
                            detector_producer: RASTER_FREE_DIALOGUE_DETECTOR,
                            detector_version: RASTER_FREE_DIALOGUE_DETECTOR_VERSION,
                            detector_input: "original_source_pixels",
                            rejected_candidates: assessment
                                .candidates
                                .iter()
                                .filter(|candidate| !candidate.accepted)
                                .cloned()
                                .collect(),
                        },
                    );
                }
                assessments_by_source.insert(source_region_id, assessment.clone());
                page_assessments.push(assessment);
            }
            page_reports.push(json!({
                "page_id": page_id,
                "detector": {
                    "producer": RASTER_FREE_DIALOGUE_DETECTOR,
                    "version": RASTER_FREE_DIALOGUE_DETECTOR_VERSION,
                    "input": "original_source_pixels",
                },
                "assessments": page_assessments,
            }));
        }

        if !decisions.is_empty() {
            let patch = edit.finish()?.with_label(
                "Detect text-safe adjacent free-dialogue target anchors from original pixels",
            );
            self.project.session().lock().await.commit(patch).await?;
        }
        *self.free_dialogue_anchors.lock() = decisions;
        *self.free_dialogue_anchor_assessments.lock() = assessments_by_source;
        Ok(page_reports)
    }

    async fn run_source_analysis(&self, control: &Control) -> Result<Invocation> {
        if self.source_analysis_completed.load(Ordering::Acquire) || self.pipeline_completed() {
            bail!("source detection and OCR analysis has already run");
        }
        koharu_ml::init()
            .await
            .context("failed to initialize the Koharu model runtime")?;
        let snapshot = self.project.session().lock().await.snapshot();
        let stop = StopToken::default();
        let watcher = tokio::spawn({
            let control = control.clone();
            let stop = stop.clone();
            async move {
                control.cancelled().await;
                stop.stop();
            }
        });
        let mut committer = SessionCommitter {
            session: self.project.session().clone(),
        };
        let result = self
            .pipeline
            .execute(
                snapshot,
                koharu_pipeline::Request {
                    operation: Operation::Through { stage: Stage::Ocr },
                    scope: Scope::Project,
                    source_language: Some(self.source_language),
                    stop,
                    progress: None,
                    inpainting_mask: None,
                },
                &mut committer,
            )
            .await;
        watcher.abort();
        let report = result.map_err(|error| anyhow!(error))?;
        if report.status == RunStatus::Stopped {
            bail!("source detection and OCR analysis was cancelled");
        }
        let ui_panel_detection = self.run_source_raster_panel_detection().await?;
        let ui_panel_anchor_binding = self.bind_deterministic_ui_panel_anchors().await?;
        let final_revision = self.project.session().lock().await.snapshot().revision();
        let inspection = self.inspect_project().await?;
        let pending_screening = decorative_sfx_disposition_candidates(&inspection);
        let pending_screening_candidates = pending_screening.len();
        *self.pending_decorative_sfx_dispositions.lock() = pending_screening;
        self.source_analysis_completed
            .store(true, Ordering::Release);
        let record = json!({
            "schema_version": 3,
            "status": "completed",
            "base_revision": report.base,
            "pipeline_final_revision": report.final_revision,
            "final_revision": final_revision,
            "completed": report.completed,
            "total": report.total,
            "elapsed_ms": report.elapsed.as_millis(),
            "ui_panel_detection": ui_panel_detection,
            "ui_panel_anchor_binding": ui_panel_anchor_binding,
            "pending_screening_candidates": pending_screening_candidates,
            "next_action": if pending_screening_candidates == 0 {
                "no screening candidates remain; inspect the source evidence, then run_pipeline"
                    .to_owned()
            } else {
                format!(
                    "inspect fresh source evidence and classify all {pending_screening_candidates} pending dialogue/free-text candidate(s) as translate, skip_difficult, or retain_required before run_pipeline"
                )
            },
        });
        self.trace_records
            .lock()
            .push(HostTraceRecord::new("source_analysis", record.clone()));
        Invocation::changed(record)
    }

    async fn classify_decorative_sfx(&self, call: &ToolCall) -> Result<Invocation> {
        if !self.source_analysis_completed.load(Ordering::Acquire) || self.pipeline_completed() {
            bail!(
                "classify_decorative_sfx is available only after source analysis and before translation"
            );
        }
        let arguments: ClassifyDecorativeSfx = arguments(call)?;
        let snapshot = self.project.session().lock().await.snapshot();
        let page_id = resolve_page_ordinal(&snapshot, arguments.page_ordinal)?.id;
        let revision = snapshot.revision();
        if Revision::new(arguments.scene_revision) != revision {
            bail!(
                "decorative-SFX classification is stale: requested revision {}, current revision {revision}",
                arguments.scene_revision
            );
        }
        let pending = self.pending_decorative_sfx_dispositions.lock().clone();
        let obligation =
            screening_obligation(&snapshot, &pending, &self.page_semantic_evidence.lock())?;
        let allowed = obligation
            .candidates
            .iter()
            .map(|candidate| candidate.element_id)
            .collect::<BTreeSet<_>>();
        for decision in &arguments.decisions {
            let element_id = entity(&decision.element)?;
            if !allowed.contains(&element_id) {
                bail!(
                    "classify_decorative_sfx element {element_id} is outside the next screening obligation: {}",
                    obligation.description()
                );
            }
        }
        let inspection = self.inspect_snapshot(snapshot.clone()).await?;
        let page = inspection
            .project
            .pages
            .iter()
            .find(|page| page.id == page_id)
            .context("classified page is missing from project inspection")?;
        let prepared = validate_decorative_sfx_classifications(
            page,
            &self.page_semantic_evidence.lock(),
            revision,
            &arguments,
        )?;

        let mut edit = snapshot.edit();
        for item in &prepared {
            let classified_role = match item.request.disposition {
                DecorativeSfxDisposition::Translate => Some(DECORATIVE_SFX_ROLE),
                DecorativeSfxDisposition::SkipDifficult => Some(SKIPPED_DIFFICULT_SFX_ROLE),
                DecorativeSfxDisposition::RetainRequired => None,
            };
            if let Some(role) = classified_role {
                edit.set(
                    item.content_id,
                    &TextRole {
                        origin: koharu_scene::Origin::User,
                        role: role.to_owned(),
                    },
                )?;
            }
            if item.request.disposition == DecorativeSfxDisposition::SkipDifficult {
                if snapshot
                    .component::<Translation>(item.content_id)?
                    .is_some()
                {
                    edit.remove::<Translation>(item.content_id)?;
                }
                edit.set(
                    item.element_id,
                    &Visibility {
                        origin: koharu_scene::Origin::User,
                        visible: false,
                        opacity: 1.0,
                    },
                )?;
            }
        }
        let patch = edit
            .finish()?
            .with_label("Classify evidence-backed decorative SFX");
        let committed = self.project.session().lock().await.commit(patch).await?;
        let decision_revision = committed.snapshot.revision();
        let decisions = prepared
            .into_iter()
            .map(|item| DecorativeSfxDecision {
                schema_version: DECORATIVE_SFX_DECISION_SCHEMA_VERSION,
                disposition: item.request.disposition,
                review_state: match item.request.disposition {
                    DecorativeSfxDisposition::Translate => "translated_decorative_sfx",
                    DecorativeSfxDisposition::SkipDifficult => "skipped_difficult_sfx",
                    DecorativeSfxDisposition::RetainRequired => {
                        "retained_required_non_decorative_sfx"
                    }
                },
                evidence_revision: revision,
                decision_revision,
                page_id,
                original_ordinal: item.original_ordinal,
                element_id: item.element_id,
                content_id: item.content_id,
                source_region_id: item.source_region_id,
                source_ocr: item.source_ocr,
                source_typography: item.source_typography,
                source_crop_blake3: item.request.source_crop_blake3,
                source_debug_label: item.request.source_debug_label,
                classifier: SfxClassifier {
                    kind: "agent_visual_semantic",
                    configured_model: self.configured_agent_model.clone(),
                    tool_call_id: call.call_id.clone(),
                },
                evidence: item.request.evidence,
                confidence: item.request.confidence,
                rationale: item.request.rationale,
                target_translation_owner: (item.request.disposition
                    != DecorativeSfxDisposition::SkipDifficult)
                    .then_some(item.content_id),
                target_render_owner: (item.request.disposition
                    != DecorativeSfxDisposition::SkipDifficult)
                    .then_some(item.element_id),
            })
            .collect::<Vec<_>>();
        {
            let mut recorded = self.decorative_sfx_decisions.lock();
            for decision in &decisions {
                recorded.insert(decision.element_id, decision.clone());
            }
        }
        let pending_dispositions = {
            let mut pending = self.pending_decorative_sfx_dispositions.lock();
            for decision in &decisions {
                pending.remove(&decision.element_id);
            }
            pending.clone()
        };
        self.record_page_mutation(page_id, decision_revision);
        *self.acceptance.lock() = None;
        *self.visual_review.lock() = None;
        self.trace_records.lock().push(HostTraceRecord::new(
            "decorative_sfx_classification",
            json!({
                "schema_version": 2,
                "scene_revision_before": revision,
                "scene_revision_after": decision_revision,
                "page_id": page_id,
                "decisions": decisions,
                "pending_screening_candidates": pending_dispositions.len(),
            }),
        ));
        let next_obligation = if pending_dispositions.is_empty() {
            None
        } else {
            Some(screening_obligation(
                &committed.snapshot,
                &pending_dispositions,
                &self.page_semantic_evidence.lock(),
            )?)
        };
        let next_action = if let Some(obligation) = next_obligation {
            format!(
                "{} screening candidate(s) remain; next screening obligation: {}",
                pending_dispositions.len(),
                obligation.description()
            )
        } else if decision_revision != revision {
            "no screening candidates remain; refresh source evidence for the new scene revision, then run_pipeline"
                .to_owned()
        } else {
            "no screening candidates remain; run_pipeline".to_owned()
        };
        Invocation::changed(json!({
            "scene_revision": decision_revision,
            "decisions": decisions,
            "pending_screening_candidates": pending_dispositions.len(),
            "next_action": next_action,
        }))
    }

    async fn verify_ui_panel_anchor(&self, call: &ToolCall) -> Result<Invocation> {
        if !self.source_analysis_completed.load(Ordering::Acquire) || self.pipeline_completed() {
            bail!(
                "verify_ui_panel_anchor is available only after source analysis and before translation"
            );
        }
        let arguments: VerifyUiPanelAnchor = arguments(call)?;
        let snapshot = self.project.session().lock().await.snapshot();
        let page_id = resolve_page_ordinal(&snapshot, arguments.page_ordinal)?.id;
        let revision = snapshot.revision();
        if Revision::new(arguments.scene_revision) != revision {
            bail!(
                "UI-panel verification is stale: requested revision {}, current revision {revision}",
                arguments.scene_revision
            );
        }
        let inspection = self.inspect_snapshot(snapshot.clone()).await?;
        let page = inspection
            .project
            .pages
            .iter()
            .find(|page| page.id == page_id)
            .context("verified UI-panel page is missing from project inspection")?;
        let prepared = validate_ui_panel_verifications(
            page,
            &self.page_semantic_evidence.lock(),
            revision,
            &arguments,
        )?;

        let mut edit = snapshot.edit();
        for item in &prepared {
            edit.set(
                item.content_id,
                &TextRole {
                    origin: koharu_scene::Origin::User,
                    role: UI_TEXT_ROLE.to_owned(),
                },
            )?;
        }
        let patch = edit
            .finish()?
            .with_label("Verify evidence-backed UI panel text anchors");
        let committed = self.project.session().lock().await.commit(patch).await?;
        let decision_revision = committed.snapshot.revision();
        let decisions = prepared
            .into_iter()
            .map(|item| UiPanelAnchorDecision {
                schema_version: 2,
                decision: "verified_required_ui_panel_text_anchor",
                evidence_revision: revision,
                decision_revision,
                page_id,
                original_ordinal: item.original_ordinal,
                element_id: item.element_id,
                content_id: item.content_id,
                source_region_id: item.source_region_id,
                source_ocr: item.source_ocr,
                source_crop_blake3: item.request.source_crop_blake3,
                source_debug_label: item.request.source_debug_label,
                panel: item.panel,
                source_containment_ratio: item.source_containment_ratio,
                source_intersection_ratio: item.source_intersection_ratio,
                classifier: UiPanelClassifier {
                    kind: "agent_visual_semantic",
                    configured_model: self.configured_agent_model.clone(),
                    tool_call_id: call.call_id.clone(),
                },
                evidence: item.request.evidence,
                confidence: item.request.confidence,
                association_reason: item.request.association_reason,
            })
            .collect::<Vec<_>>();
        {
            let mut recorded = self.ui_panel_anchors.lock();
            for decision in &decisions {
                recorded.insert(decision.element_id, decision.clone());
            }
        }
        self.record_page_mutation(page_id, decision_revision);
        *self.acceptance.lock() = None;
        *self.visual_review.lock() = None;
        self.trace_records.lock().push(HostTraceRecord::new(
            "ui_panel_anchor_verification",
            json!({
                "schema_version": 2,
                "scene_revision_before": revision,
                "scene_revision_after": decision_revision,
                "page_id": page_id,
                "decisions": decisions,
            }),
        ));
        let pending_screening_candidates = self.pending_decorative_sfx_dispositions.lock().len();
        Invocation::changed(json!({
            "scene_revision": decision_revision,
            "decisions": decisions,
            "pending_screening_candidates": pending_screening_candidates,
            "next_action": if pending_screening_candidates == 0 {
                "no screening candidates remain; refresh source evidence for the new scene revision, then run_pipeline"
                    .to_owned()
            } else {
                format!(
                    "{pending_screening_candidates} screening candidate(s) remain; refresh source evidence and classify each before run_pipeline"
                )
            },
        }))
    }

    async fn run_pipeline(&self, control: &Control) -> Result<Invocation> {
        if self.pipeline_completed() {
            bail!("the complete pipeline has already run; use revise_element for reviewed repairs");
        }
        if !self.source_analysis_completed.load(Ordering::Acquire) {
            bail!("run_source_analysis and source-evidence review must precede translation");
        }
        let pending_dispositions = self.pending_decorative_sfx_dispositions.lock().len();
        if pending_dispositions > 0 {
            bail!(
                "run_pipeline requires an explicit evidence-backed translate, skip_difficult, or retain_required disposition for all {pending_dispositions} pending detector-backed uncontained dialogue/free-text screening candidate(s)"
            );
        }
        koharu_ml::init()
            .await
            .context("failed to initialize the Koharu model runtime")?;
        let pages = self
            .project
            .session()
            .lock()
            .await
            .snapshot()
            .pages()
            .map(|page| page.id())
            .collect::<Vec<_>>();
        for page in pages {
            let snapshot = self.project.session().lock().await.snapshot();
            let (patch, report) =
                koharu_pipeline::prepare_translation_inputs(&snapshot, page, self.target_language)?;
            if report.adjusted() {
                self.project.session().lock().await.commit(patch).await?;
            }
            self.trace_records.lock().push(HostTraceRecord::new(
                "pipeline_preprocessing",
                serde_json::to_value(report)?,
            ));
        }
        let snapshot = self.project.session().lock().await.snapshot();
        let page_labels = snapshot
            .pages()
            .map(|page| Ok((page.id(), page.page()?.label)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let stages = Arc::new(SyncMutex::new(BTreeMap::<
            (EntityId, Stage),
            MutableStageTelemetry,
        >::new()));
        for page in page_labels.keys().copied() {
            for stage in Stage::ALL {
                stages.lock().insert(
                    (page, stage),
                    MutableStageTelemetry {
                        status: StageTelemetryStatus::NotRun,
                        model: None,
                        elapsed_ms: None,
                    },
                );
            }
        }
        let progress = {
            let stages = Arc::clone(&stages);
            let trace_records = Arc::clone(&self.trace_records);
            Arc::new(move |event: Progress| {
                if let Progress::Preprocessed { report } = &event {
                    trace_records.lock().push(HostTraceRecord::new(
                        "pipeline_preprocessing",
                        serde_json::to_value(report)
                            .expect("pipeline preprocessing report must serialize"),
                    ));
                }
                update_stage_telemetry(&stages, event);
            })
        };
        let stop = StopToken::default();
        let watcher = tokio::spawn({
            let control = control.clone();
            let stop = stop.clone();
            async move {
                control.cancelled().await;
                stop.stop();
            }
        });
        let mut committer = SessionCommitter {
            session: self.project.session().clone(),
        };
        let result = self
            .pipeline
            .execute(
                snapshot,
                koharu_pipeline::Request {
                    operation: Operation::Stages {
                        stages: vec![Stage::Translation, Stage::Inpainting],
                    },
                    scope: Scope::Project,
                    source_language: Some(self.source_language),
                    stop,
                    progress: Some(progress),
                    inpainting_mask: None,
                },
                &mut committer,
            )
            .await;
        watcher.abort();
        let failed_stage = result.as_ref().err().and_then(|error| error.stage);
        let failure_kind = result
            .as_ref()
            .err()
            .map(|error| format!("{:?}", error.kind).to_lowercase());
        let failure = result.as_ref().err().map(ToString::to_string);
        let stopped = result
            .as_ref()
            .is_ok_and(|report| report.status == RunStatus::Stopped);
        if failure.is_some() || stopped {
            for ((_, stage_kind), stage) in stages.lock().iter_mut() {
                if matches!(
                    stage.status,
                    StageTelemetryStatus::Loading | StageTelemetryStatus::Running
                ) {
                    stage.status = if failure.is_some()
                        && failed_stage.is_none_or(|failed| failed == *stage_kind)
                    {
                        StageTelemetryStatus::Failed
                    } else {
                        StageTelemetryStatus::Aborted
                    };
                }
            }
        }
        let status = match result.as_ref().map(|report| report.status) {
            Ok(RunStatus::Completed) => PipelineRunTelemetryStatus::Completed,
            Ok(RunStatus::Stopped) => PipelineRunTelemetryStatus::Stopped,
            Err(_) => PipelineRunTelemetryStatus::Failed,
        };
        let inspection = self.inspect_project().await?;
        let mut telemetry = build_pipeline_telemetry(
            status,
            failed_stage,
            failure_kind,
            failure.clone(),
            &page_labels,
            &stages.lock(),
            &inspection,
            self.target_language.tag(),
        );
        let semantic_failure = result
            .as_ref()
            .is_ok_and(|report| report.status == RunStatus::Completed)
            .then(|| validate_required_translations(&telemetry.pages).err())
            .flatten()
            .map(|error| error.to_string());
        if let Some(failure) = semantic_failure.as_ref() {
            telemetry.status = PipelineRunTelemetryStatus::Failed;
            telemetry.failure_stage = Some(Stage::Translation);
            telemetry.failure_kind = Some("invalid_output".to_owned());
            telemetry.failure = Some(format!(
                "pipeline required-translation validation failed: {failure}"
            ));
        }
        *self.telemetry.lock() = Some(telemetry.clone());
        self.trace_records.lock().push(HostTraceRecord::new(
            "pipeline_telemetry",
            serde_json::to_value(&telemetry)?,
        ));
        let report = result.map_err(|error| anyhow!(error))?;
        if report.status == RunStatus::Stopped {
            bail!("pipeline processing was cancelled");
        }
        if let Some(failure) = semantic_failure {
            bail!("pipeline required-translation validation failed: {failure}");
        }
        let free_dialogue_anchor_detection = self
            .run_source_raster_free_dialogue_anchor_detection()
            .await?;
        let free_dialogue_revision = self.project.session().lock().await.snapshot().revision();
        self.trace_records.lock().push(HostTraceRecord::new(
            "free_dialogue_anchor_detection",
            json!({
                "schema_version": 1,
                "scene_revision": free_dialogue_revision,
                "pages": &free_dialogue_anchor_detection,
            }),
        ));
        mark_pipeline_completed(&self.pipeline_completed, &telemetry.pages)?;
        let final_revision = self.project.session().lock().await.snapshot().revision();
        *self.page_reviews.lock() =
            PageReviewState::at_revision(page_labels.keys().copied(), final_revision);
        Invocation::changed(json!({
            "base_revision": report.base,
            "pipeline_final_revision": report.final_revision,
            "final_revision": final_revision,
            "completed": report.completed,
            "total": report.total,
            "elapsed_ms": report.elapsed.as_millis(),
            "telemetry": telemetry,
            "free_dialogue_anchor_detection": free_dialogue_anchor_detection,
        }))
    }
}

fn configure_pipeline_for_harness(
    config: &mut koharu_pipeline::PipelineConfig,
    source_language: Language,
    target_language: Language,
) {
    if source_language == Language::English {
        config.ocr = OcrModel::HayaiOcr;
    }
    config.translation.target_language = target_language;
}

#[derive(Clone, Debug)]
struct RepairToolConstraint {
    name: &'static str,
    operation: Option<&'static str>,
    element: Option<EntityId>,
    logical_group: Option<EntityId>,
    page: Option<PageOrdinal>,
    revision: Option<Revision>,
    preview_id: Option<String>,
    allowed_fields: Vec<RepairField>,
}

impl RepairToolConstraint {
    fn commit(name: &'static str, preview_id: &str, element: EntityId) -> RepairToolConstraint {
        Self {
            name,
            operation: None,
            element: Some(element),
            logical_group: None,
            page: None,
            revision: None,
            preview_id: Some(preview_id.to_owned()),
            allowed_fields: Vec::new(),
        }
    }
}

fn store_pending_text_layout(
    pending: &mut BTreeMap<String, PreparedTextLayout>,
    preview_id: String,
    prepared: PreparedTextLayout,
) {
    pending.clear();
    pending.insert(preview_id, prepared);
}

fn pending_text_layout_commit(
    review: &VisualReviewRecord,
    pending: &BTreeMap<String, PreparedTextLayout>,
) -> Option<RepairToolConstraint> {
    let failure = review.deterministic_repair_plan.blocking_failures.first()?;
    let (preview_id, prepared) = (pending.len() == 1).then(|| pending.first_key_value())??;
    let preview = &prepared.preview;
    pending_preview_commit(
        review,
        failure,
        preview_id,
        &preview.preview_id,
        "preview_text_layout",
        "commit_text_layout",
        preview.operation,
        preview.element_id,
        preview.base_revision,
        &prepared.evidence,
        preview.deterministic_constraints_satisfied,
        preview.mutation_committed,
        validate_text_layout_plan(Some(failure), &preview.options).is_ok(),
    )
}

fn pending_compact_translation_commit(
    review: &VisualReviewRecord,
    pending: &BTreeMap<String, PreparedCompactTranslation>,
    observations: &PageSemanticEvidenceState,
) -> Option<RepairToolConstraint> {
    let failure = review.deterministic_repair_plan.blocking_failures.first()?;
    let (preview_id, prepared) = (pending.len() == 1).then(|| pending.first_key_value())??;
    let preview = &prepared.preview;
    pending_preview_commit(
        review,
        failure,
        preview_id,
        &preview.preview_id,
        "preview_compact_translation",
        "commit_compact_translation",
        preview.operation,
        preview.primary_render_element_id,
        preview.base_revision,
        &prepared.evidence,
        preview.deterministic_constraints_satisfied,
        preview.mutation_committed,
        failure.logical_group_id == Some(preview.logical_group_id)
            && failure.member_ordinal_ids == preview.member_ordinal_ids
            && prepared.member_ordinal_ids == preview.member_ordinal_ids
            && preview.candidate_visible_grapheme_units > 0
            && preview.candidate_visible_grapheme_units < preview.current_visible_grapheme_units
            && validate_compact_translation_plan(
                failure,
                preview.primary_render_element_id,
                preview.logical_group_id,
            )
            .is_ok()
            && validate_prepared_compact_evidence(
                observations,
                review.scene_revision,
                &prepared.evidence,
                &prepared.member_ordinal_ids,
            )
            .is_ok()
            && validate_compact_translation_change(&prepared.before_state, &prepared.after_state)
                .is_ok(),
    )
}

#[allow(clippy::too_many_arguments)]
fn pending_preview_commit(
    review: &VisualReviewRecord,
    failure: &DeterministicRepairFailure,
    pending_preview_id: &str,
    preview_id: &str,
    preview_tool: &'static str,
    commit_tool: &'static str,
    operation: &'static str,
    element: EntityId,
    base_revision: Revision,
    evidence: &RevisionEvidence,
    deterministic_constraints_satisfied: bool,
    mutation_committed: bool,
    preview_kind_is_valid: bool,
) -> Option<RepairToolConstraint> {
    let action = failure.next_action.as_ref()?;
    (pending_preview_id == preview_id
        && !preview_id.trim().is_empty()
        && action.tool == preview_tool
        && action.operation == operation
        && action.element == element
        && action.required_evidence_revision == review.scene_revision
        && failure.required_evidence_revision == review.scene_revision
        && failure.primary_render_element_id.or(failure.element_id) == Some(element)
        && base_revision == review.scene_revision
        && evidence.revision() == review.scene_revision
        && deterministic_constraints_satisfied
        && !mutation_committed
        && preview_kind_is_valid)
        .then(|| RepairToolConstraint::commit(commit_tool, preview_id, element))
}

fn matching_source_evidence(evidence: &PageSemanticEvidenceState, page: EntityId) -> bool {
    let Some(evidence) = evidence.page(page) else {
        return false;
    };
    matches!(
        (&evidence.source_dossier, &evidence.source_debug_artifact),
        (Some(dossier), Some(debug))
            if dossier.revision == debug.revision && dossier.page_id == debug.page_id
    )
}

fn complete_page_evidence(
    evidence: &PageSemanticEvidenceState,
    page: EntityId,
) -> Option<(Revision, EntityId)> {
    let evidence = evidence.page(page)?;
    let source = evidence.source_dossier.as_ref()?;
    let source_debug = evidence.source_debug_artifact.as_ref()?;
    let dossier = evidence.dossier.as_ref()?;
    let debug = evidence.debug_artifact.as_ref()?;
    (source.revision == source_debug.revision
        && source.revision == dossier.revision
        && source.revision == debug.revision
        && source.page_id == source_debug.page_id
        && source.page_id == dossier.page_id
        && source.page_id == debug.page_id)
        .then_some((source.revision, source.page_id))
}

fn constrain_repair_tool(tool: &mut Tool, repair: &RepairToolConstraint) {
    if repair.name == "preview_text_layout"
        && let Some(operation) = repair.operation
    {
        let required_options =
            if operation == "controlled_source_bound_vertical_interjection_fallback" {
                "SourceBoundInterjectionFallbackOptions"
            } else {
                "ControlledTextLayoutOptions"
            };
        if let Some(variants) = tool
            .parameters
            .pointer_mut("/$defs/TextLayoutOptions/anyOf")
            .and_then(Value::as_array_mut)
        {
            variants.retain(|variant| {
                variant
                    .get("$ref")
                    .and_then(Value::as_str)
                    .is_some_and(|reference| reference.ends_with(required_options))
            });
        }
        tool.description
            .push_str(&format!(" Active deterministic operation: {operation}."));
        if required_options == "SourceBoundInterjectionFallbackOptions" {
            tool.description.push_str(
                " Use exactly the recorded native_vertical fallback options object; it is exposed only after a recorded no-adjacent-candidate-passed outcome.",
            );
        } else {
            tool.description.push_str(
                " Use controlled layout options only; native fallback is not eligible for this operation.",
            );
        }
    }
    if let Some(element) = repair.element {
        match repair.name {
            "revise_page_translation" => {
                constrain_pointer_const(
                    &mut tool.parameters,
                    "/$defs/PageTranslationEdit/properties/element",
                    json!(element.to_string()),
                );
                constrain_object_keyword(
                    &mut tool.parameters,
                    "/properties/edits",
                    "minItems",
                    json!(1),
                );
                constrain_object_keyword(
                    &mut tool.parameters,
                    "/properties/edits",
                    "maxItems",
                    json!(1),
                );
                let source_allowed = repair.allowed_fields.contains(&RepairField::SourceText);
                let translation_allowed = repair
                    .allowed_fields
                    .contains(&RepairField::TranslationText);
                if !source_allowed {
                    constrain_pointer_schema_false(
                        &mut tool.parameters,
                        "/$defs/PageTranslationEdit/properties/source",
                    );
                }
                if !translation_allowed {
                    constrain_pointer_schema_false(
                        &mut tool.parameters,
                        "/$defs/PageTranslationEdit/properties/translation",
                    );
                }
            }
            "preview_compact_translation" => {
                constrain_string_property(
                    &mut tool.parameters,
                    "primary_render_element",
                    element.to_string(),
                );
            }
            name if name.starts_with("preview_") => {
                constrain_string_property(&mut tool.parameters, "element", element.to_string());
            }
            _ => {}
        }
        tool.description
            .push_str(&format!(" This turn is restricted to element {element}."));
    }
    if let Some(group) = repair.logical_group
        && repair.name == "preview_compact_translation"
    {
        constrain_string_property(&mut tool.parameters, "logical_group", group.to_string());
    }
    if let Some(page) = repair.page {
        constrain_usize_property(&mut tool.parameters, "page_ordinal", page.get());
    }
    if let Some(revision) = repair.revision {
        constrain_pointer_const(
            &mut tool.parameters,
            "/$defs/PageTranslationEvidenceReference/properties/scene_revision",
            json!(revision.get()),
        );
    }
    if let Some(preview_id) = repair.preview_id.as_ref() {
        constrain_string_property(&mut tool.parameters, "preview_id", preview_id.clone());
        tool.description
            .push_str(" Only the current host-issued preview ID is accepted.");
    }
}

fn constrain_string_property(schema: &mut Value, property: &str, value: String) {
    constrain_pointer_const(
        schema,
        &format!("/properties/{property}"),
        Value::String(value),
    );
}

fn constrain_u64_property(schema: &mut Value, property: &str, value: u64) {
    constrain_pointer_const(schema, &format!("/properties/{property}"), json!(value));
}

fn constrain_usize_property(schema: &mut Value, property: &str, value: usize) {
    constrain_pointer_const(schema, &format!("/properties/{property}"), json!(value));
}

fn constrain_pointer_const(schema: &mut Value, pointer: &str, value: Value) {
    let Some(target) = schema.pointer_mut(pointer) else {
        return;
    };
    if let Some(object) = target.as_object_mut() {
        object.insert("const".to_owned(), value);
    } else {
        *target = json!({ "const": value });
    }
}

fn constrain_object_keyword(schema: &mut Value, pointer: &str, keyword: &str, value: Value) {
    if let Some(object) = schema.pointer_mut(pointer).and_then(Value::as_object_mut) {
        object.insert(keyword.to_owned(), value);
    }
}

fn constrain_pointer_schema_false(schema: &mut Value, pointer: &str) {
    if let Some(target) = schema.pointer_mut(pointer) {
        *target = Value::Bool(false);
    }
}

#[async_trait]
impl Host for HarnessHost {
    async fn context(&self) -> Result<Value> {
        self.project_context().await
    }

    fn tools(&self) -> Vec<Tool> {
        let (phase, tools) = self.workflow_surface();
        self.trace_records.lock().push(HostTraceRecord::new(
            "workflow_tool_surface",
            json!({
                "schema_version": 1,
                "workflow_phase": phase,
                "exposed_tool_count": tools.len(),
                "exposed_tools": tools.iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>(),
            }),
        ));
        tools
    }

    async fn invoke(&self, call: ToolCall, control: &Control) -> Result<Invocation> {
        let (phase, tools) = self.workflow_surface();
        let semantic_review_obligation = self.next_semantic_review_obligation()?;
        if phase == WorkflowPhase::SourceEvidence
            && is_source_screening_tool(&call.name)
            && let Some(obligation) = self.next_screening_obligation()?
        {
            validate_source_screening_tool_page(&call, &obligation)?;
        }
        if let Some(obligation) = semantic_review_obligation
            && is_semantic_review_tool(&call.name)
        {
            validate_semantic_review_tool(&call, obligation)?;
        }
        ensure_tool_available(phase, &tools, &call.name)?;
        let mut invocation = match call.name.as_str() {
            "inspect_project" => {
                let _: InspectProject = arguments(&call)?;
                let inspection = self.inspect_project().await?;
                *self.last_explicit_observation.lock() =
                    Some(RevisionEvidence::ProjectInspection {
                        revision: inspection.project.revision,
                    });
                Invocation::read(serde_json::to_value(inspection)?)
            }
            "view_page" => {
                let arguments: ViewPage = arguments(&call)?;
                let (page, label, snapshot) = {
                    let session = self.project.session().lock().await;
                    let snapshot = session.snapshot();
                    let page = resolve_page_ordinal(&snapshot, arguments.page_ordinal)?;
                    let label = snapshot.page(page.id)?.page()?.label;
                    (page, label, snapshot)
                };
                let bytes =
                    rendered_preview(&self.renderer, self.rasterizer().await?, &snapshot, page.id)
                        .await?;
                let provenance = ToolImageProvenance::ContentHash {
                    algorithm: "blake3",
                    digest: blake3::hash(&bytes).to_hex().to_string(),
                    media_type: "image/webp".to_owned(),
                    byte_length: bytes.len(),
                };
                *self.last_explicit_observation.lock() =
                    Some(RevisionEvidence::RenderedPageInspection {
                        revision: snapshot.revision(),
                        page_id: page.id,
                    });
                Ok(Invocation::read(json!({
                    "page_ordinal": page.ordinal.get(),
                    "page_id": page.id,
                    "label": label,
                }))?
                .with_image(
                    format!(
                        "Rendered page ordinal {}: {label} ({})",
                        page.ordinal.get(),
                        page.id
                    ),
                    format!("data:image/webp;base64,{}", STANDARD.encode(bytes)),
                    provenance,
                ))
            }
            "inspect_source_evidence" => self.inspect_source_evidence(&call).await,
            "view_page_source_debug" => self.view_page_source_debug(&call).await,
            "run_source_analysis" => {
                let _: RunSourceAnalysis = arguments(&call)?;
                self.run_source_analysis(control).await
            }
            "classify_decorative_sfx" => self.classify_decorative_sfx(&call).await,
            "verify_ui_panel_anchor" => self.verify_ui_panel_anchor(&call).await,
            "inspect_page_evidence" => self.inspect_page_evidence(&call).await,
            "run_pipeline" => {
                let _: RunPipeline = arguments(&call)?;
                self.run_pipeline(control).await
            }
            "review_pages" => {
                let _: ReviewPages = arguments(&call)?;
                if !self.pipeline_completed() {
                    bail!("the complete Koharu pipeline must finish before visual review");
                }
                let review = self.record_review_and_execute_host_repairs().await?;
                self.reset_unresolved_semantic_evidence(&review);
                let mut output = serde_json::to_value(review)?;
                if let Some(object) = output.as_object_mut() {
                    object.insert(
                        "host_repair_stop_diagnostic".to_owned(),
                        serde_json::to_value(self.repair_stop.lock().clone())?,
                    );
                }
                Invocation::changed(output)
            }
            "submit_visual_semantic_review" => self.submit_visual_semantic_review(&call).await,
            "revise_element" => self.revise_element(&call).await,
            "revise_page_translation" => self.revise_page_translation(&call).await,
            "preview_text_layout" => self.preview_text_layout(&call).await,
            "commit_text_layout" => self.commit_text_layout(&call).await,
            "preview_compact_translation" => self.revise_compact_translation(&call).await,
            "commit_compact_translation" => self.commit_compact_translation(&call).await,
            "preview_increase_text_safe_padding" => {
                self.preview_text_safe_layout_repair(&call).await
            }
            "commit_text_safe_layout_repair" => self.commit_text_safe_layout_repair(&call).await,
            "export_pages" => {
                let _: ExportPages = arguments(&call)?;
                if !self.pipeline_completed() {
                    self.trace_records.lock().push(HostTraceRecord::new(
                        "export_telemetry",
                        json!({
                            "schema_version": 1,
                            "status": "rejected",
                            "format": self.output_format.as_str(),
                            "files": [],
                            "error": "complete pipeline has not finished",
                        }),
                    ));
                    bail!("the complete Koharu pipeline must finish before export");
                }
                let acceptance = self.record_acceptance().await?;
                if !acceptance.accepted {
                    self.trace_records.lock().push(HostTraceRecord::new(
                        "export_telemetry",
                        json!({
                            "schema_version": 1,
                            "status": "rejected",
                            "format": self.output_format.as_str(),
                            "files": [],
                            "error": "acceptance criteria failed",
                        }),
                    ));
                    bail!(
                        "export rejected by acceptance criteria: {} rejection reason(s)",
                        acceptance.rejection_reasons.len()
                    );
                }
                let Some(visual_review) = self.visual_review_record() else {
                    self.trace_records.lock().push(HostTraceRecord::new(
                        "export_telemetry",
                        json!({
                            "schema_version": 1,
                            "status": "rejected",
                            "format": self.output_format.as_str(),
                            "files": [],
                            "error": "visual review has not run",
                        }),
                    ));
                    bail!(
                        "export requires review_pages and an accepted visual/semantic review decision"
                    );
                };
                let snapshot = self.project.session().lock().await.snapshot();
                if let Err(error) = validate_visual_review_export_precondition(
                    &visual_review,
                    &self.page_reviews.lock(),
                    snapshot.revision(),
                ) {
                    self.trace_records.lock().push(HostTraceRecord::new(
                        "export_telemetry",
                        json!({
                            "schema_version": 1,
                            "status": "rejected",
                            "format": self.output_format.as_str(),
                            "files": [],
                            "error": error.to_string(),
                            "visual_review": visual_review,
                        }),
                    ));
                    return Err(error);
                }
                let directory = self.output_directory.clone();
                let format = self.output_format.into();
                let exported = export_pages(
                    self.renderer.clone(),
                    self.rasterizer().await?,
                    snapshot,
                    Vec::new(),
                    format,
                    directory,
                )
                .await;
                let paths = match exported {
                    Ok(paths) => {
                        self.trace_records.lock().push(HostTraceRecord::new(
                            "export_telemetry",
                            json!({
                                "schema_version": 1,
                                "status": "completed",
                                "format": self.output_format.as_str(),
                                "files": paths_to_values(&paths)?,
                                "error": null,
                            }),
                        ));
                        paths
                    }
                    Err(error) => {
                        self.trace_records.lock().push(HostTraceRecord::new(
                            "export_telemetry",
                            json!({
                                "schema_version": 1,
                                "status": "failed",
                                "format": self.output_format.as_str(),
                                "files": [],
                                "error": format!("{error:#}"),
                            }),
                        ));
                        return Err(error);
                    }
                };
                *self.outputs.lock() = paths.clone();
                Invocation::changed(json!({
                    "files": paths_to_values(&paths)?,
                    "acceptance": acceptance,
                    "visual_review": visual_review,
                }))
            }
            name => bail!("unknown Koharu harness tool {name}"),
        }?;
        if (call.name == "review_pages" || is_semantic_review_tool(&call.name))
            && let Some(object) = invocation.value.as_object_mut()
        {
            let next_obligation = self.next_semantic_review_obligation()?;
            if semantic_review_obligation.is_some() || call.name == "review_pages" {
                object.remove("next_actions");
                object.insert(
                    "next_action".to_owned(),
                    next_obligation.map_or_else(
                        || {
                            self.visual_review_record()
                                .is_some_and(|review| review.accepted())
                                .then_some(Value::String("export_pages".to_owned()))
                                .unwrap_or(Value::Null)
                        },
                        |obligation| Value::String(obligation.required_stage.as_str().to_owned()),
                    ),
                );
            }
            object.insert(
                "semantic_review_obligation".to_owned(),
                next_obligation
                    .map(SemanticReviewObligation::value)
                    .unwrap_or(Value::Null),
            );
        }
        Ok(invocation)
    }

    fn take_trace_records(&self) -> Vec<HostTraceRecord> {
        std::mem::take(&mut *self.trace_records.lock())
    }

    async fn completion(&self) -> Result<HostCompletion> {
        let (phase, tools) = self.workflow_surface();
        let scene_revision = self.project.session().lock().await.snapshot().revision();
        let progress_marker = self.continuation_progress_marker(phase, scene_revision);
        let source_completion = if phase == WorkflowPhase::SourceEvidence {
            self.next_screening_obligation()?
                .map(|obligation| HostCompletion::Continue {
                    phase: phase.as_str().to_owned(),
                    exposed_tools: tools.iter().map(|tool| tool.name.clone()).collect(),
                    reason: format!(
                        "pending source screening requires the next obligation: {}",
                        obligation.description()
                    ),
                    progress_marker: json!({
                        "scene_revision": scene_revision.get(),
                        "workflow_phase": phase.as_str(),
                        "page_ordinal": obligation.page.ordinal.get(),
                        "page_id": obligation.page.id,
                        "element_id": obligation.next_candidate().element_id,
                        "original_ordinal": obligation.next_candidate().original_ordinal,
                        "required_evidence_stage": obligation.required_evidence_stage.as_str(),
                    })
                    .to_string(),
                })
        } else {
            None
        };
        let semantic_completion = if matches!(
            phase,
            WorkflowPhase::PageEvidence | WorkflowPhase::SemanticReview
        ) {
            self.next_semantic_review_obligation()?
                .map(|obligation| HostCompletion::Continue {
                    phase: phase.as_str().to_owned(),
                    exposed_tools: tools.iter().map(|tool| tool.name.clone()).collect(),
                    reason: format!(
                        "pending visual/semantic review requires the next obligation: {}",
                        obligation.description()
                    ),
                    progress_marker: json!({
                        "scene_revision": scene_revision.get(),
                        "workflow_phase": phase.as_str(),
                        "page_ordinal": obligation.page.ordinal.get(),
                        "page_id": obligation.page.id,
                        "required_stage": obligation.required_stage.as_str(),
                    })
                    .to_string(),
                })
        } else {
            None
        };
        let completion = match source_completion {
            Some(completion) => Some(completion),
            None => match semantic_completion {
                Some(completion) => Some(completion),
                None => actionable_completion(phase, &tools, progress_marker, || {
                    self.continuation_reason(phase)
                })?,
            },
        };
        if let Some(completion) = completion {
            if let HostCompletion::Continue {
                phase,
                exposed_tools,
                reason,
                progress_marker,
            } = &completion
            {
                self.trace_records.lock().push(HostTraceRecord::new(
                    "host_completion",
                    json!({
                        "schema_version": 1,
                        "status": "continue",
                        "workflow_phase": phase,
                        "exposed_tools": exposed_tools,
                        "reason": reason,
                        "progress_marker": progress_marker,
                    }),
                ));
            }
            return Ok(completion);
        }
        if !self.pipeline_completed() {
            bail!("agent completed without running the complete Koharu pipeline");
        }
        let acceptance = match self.acceptance_record() {
            Some(record) => record,
            None => self.record_acceptance().await?,
        };
        let visual_review = match self.visual_review_record() {
            Some(record) => record,
            None => self.record_visual_review().await?,
        };
        if !acceptance.accepted {
            bail!(
                "agent completed with {} acceptance rejection reason(s)",
                acceptance.rejection_reasons.len()
            );
        }
        let current_revision = self.project.session().lock().await.snapshot().revision();
        validate_visual_review_export_precondition(
            &visual_review,
            &self.page_reviews.lock(),
            current_revision,
        )
        .context("agent completed without an accepted visual/semantic review")?;
        let outputs = self.outputs.lock();
        if outputs.len() != acceptance.pages.len() {
            bail!(
                "agent completed without exporting every accepted page: {} exports for {} pages",
                outputs.len(),
                acceptance.pages.len()
            );
        }
        if let Some(path) = outputs.iter().find(|path| !path.is_file()) {
            bail!("agent completed with a missing export: {}", path.display());
        }
        self.trace_records.lock().push(HostTraceRecord::new(
            "host_completion",
            json!({
                "schema_version": 1,
                "status": "completed",
                "workflow_phase": phase,
                "exposed_tools": tools.iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>(),
                "reason": null,
            }),
        ));
        Ok(HostCompletion::Completed)
    }
}

fn tool_definitions() -> &'static Vec<Tool> {
    static TOOLS: OnceLock<Vec<Tool>> = OnceLock::new();
    TOOLS.get_or_init(|| {
        vec![
            definition::<InspectProject>(
                "inspect_project",
                "Authorized only in a terminal review-failure diagnostic phase. Returns a bounded project index with revision/page/element/role/text, concise bounds, rejection and active repair-plan bindings, plus a path and BLAKE3 reference to the complete diagnostic artifact.",
            ),
            definition::<ViewPage>(
                "view_page",
                "Render and inspect one imported page when visual inspection is necessary.",
            ),
            definition::<RunSourceAnalysis>(
                "run_source_analysis",
                "Run detector and OCR analysis for every imported page without translating or inpainting, then deterministically test original source pixels around each detected text box for a closed, high-confidence rectangular panel with a distinct safe interior and minimum text room. Exact unambiguous detector/source pairs are host-bound to the required UI role and panel safe interior before translation; ambiguous, dialogue, caption, SFX, non-free-text, and non-source-bound elements remain untouched. The trace records accepted measurements, bindings, rejection reasons, and the exact pending uncontained dialogue/free-text screening count. This required first phase exposes authoritative source crops so each pending candidate can be classified as decorative SFX or retained as ordinary required text before translation.",
            ),
            definition::<InspectSourceEvidence>(
                "inspect_source_evidence",
                "After source analysis, inspect authoritative original pixels for source screening. Writes a read-only page-scoped dossier and stable full-page/original-crop artifacts for every ordinal, including text role, required versus skipped_difficult_sfx state, verified UI-panel evidence, source/candidate bounds, and source-raster measurements. Original pixels are authoritative and OCR is fallible. Pair this source-screening API with view_page_source_debug before translation.",
            ),
            definition::<ViewPageSourceDebug>(
                "view_page_source_debug",
                "Render a headless read-only overlay on the original imported page, separately from translated debug output. Uses deterministic reading-order IDs and draws source regions, panel candidates, verified UI anchors, and source-raster-proven adjacent free-dialogue anchors with a legend. Review moved free dialogue against the original speaker/reaction and visual attribution. Returns stable scratch evidence without changing project or export state.",
            ),
            definition::<ClassifyDecorativeSfx>(
                "classify_decorative_sfx",
                "Before translation, resolve detector-backed required, valid-geometry, uncontained dialogue/free-text screening candidates using fresh inspect_source_evidence and view_page_source_debug bindings. Every decision must match the exact original ordinal/ID/crop/debug label, use confidence >= 0.90, and separately assess visual form, page function, legibility/translation value, and required-content role. Use translate only for positively evidenced legible decorative SFX, skip_difficult only for positively evidenced difficult decorative SFX, and retain_required when the pixels establish ordinary dialogue/free text rather than decorative SFX. retain_required preserves the existing role, required state, visibility, and ordinary translation/inpaint/acceptance path. Container-linked or ineligible roles, ambiguous evidence, stale evidence, duplicate targets, and text-token evidence are rejected. The live description reports the remaining pending count; run_pipeline stays unavailable until it reaches zero.",
            ),
            definition::<VerifyUiPanelAnchor>(
                "verify_ui_panel_anchor",
                "Before translation, positively classify required detector-backed free text as UI and bind it to one explicit source-raster-verified panel/screen candidate from the current source dossier/debug overlay. The exact ordinal, source crop, debug label, existing panel region ID, UI role/function, visible finite screen evidence, source-to-panel relation, text-safe interior, detection/visual provenance, confidence >= 0.90, and reason are required in addition to the immutable closed-contour, distinct-interior, safe-room, source-containment, and detector/version record. Model-only panels, dialogue, SFX, captions/general free text, weak association, non-contained source text, inferred geometry, and nearest-panel selection are rejected. Verification records evidence only; target placement remains source-bound until an exact preview_text_layout candidate succeeds and is committed.",
            ),
            definition::<InspectPageEvidence>(
                "inspect_page_evidence",
                "Atomically produce the source dossier/debug and translated dossier/debug artifacts for one exact translated page ordinal. Return and bind all four exact current-revision BLAKE3 digests for direct semantic submission, with accepted earlier pages as read-only context.",
            ),
            definition::<RunPipeline>(
                "run_pipeline",
                "After run_source_analysis and explicit evidence-backed screening decisions for every pending uncontained dialogue/free-text candidate, run preprocessing plus translation and inpainting for every page. Legible decorative SFX, retain_required dialogue/free text, captions, UI, and ambiguous required regions remain on the ordinary required path; only recorded skipped_difficult_sfx elements are excluded.",
            ),
            definition::<ReviewPages>(
                "review_pages",
                "After inspect_page_evidence, create the current-revision review bundle containing each actual original page, translated rendered preview, all semantic source elements (including skipped_difficult_sfx), and deterministic acceptance. Host-recorded exact target migrations are previewed, globally validated, committed, and deterministically re-reviewed only when a blocking page-level safety failure authorizes them; diagnostic font, glyph, anchor, and contour measurements never trigger repair. A failed host candidate records and surfaces a terminal diagnostic without mutation. Semantic wording revision remains model-facing. With a configured command judge, preserve its independent decision. Without one, return pending_agent_review; then inspect_page_evidence and submit_visual_semantic_review for each exact page/revision. No semantic decision overrides deterministic export gates.",
            ),
            definition::<SubmitVisualSemanticReview>(
                "submit_visual_semantic_review",
                "Submit one narrow structured page judgment only after inspect_page_evidence for the exact current semantic-review obligation. Bind the exact current scene_revision and all four bundled BLAKE3 digests for that same page. Compare the authoritative original full page/crops and source-debug overlay with the translated dossier/render/debug overlay; explicitly judge source pixels, translated meaning, target-language naturalness, reading order, omissions/duplicates, typography/layout, and whether every skipped item is truly difficult decorative SFX with no required content skipped. For every committed compact translation, compacted_translation_reviews must bind every ordered original group-member crop and explicitly judge meaning fidelity and natural Korean; drift must be reported in both compact and page issues and blocks export. Use an empty list when there was no compaction. Never infer semantic or visual acceptance from deterministic geometry. accepted=true requires all seven booleans true, an issue-free nonempty summary, and fresh matching evidence. Rejection requires concrete retained issues and blocks export. This tool cannot replace or bypass a configured command judge.",
            ),
            definition::<ReviseElement>(
                "revise_element",
                "Correct one element after review. Prefer revise_page_translation whenever wording, continuity, register, tone, punctuation, or unwanted parentheses span multiple balloons. Font, glyph, anchor, and contour-clearance diagnostics do not authorize repair. When a deterministic plan permits this tool, target its first unresolved element and only its allowed fields.",
            ),
            definition::<RevisePageTranslation>(
                "revise_page_translation",
                "Atomically revise page-level source/translation semantics only after inspect_page_evidence has produced all four fresh matching evidence artifacts for the exact current page revision. Evidence must include that revision and all four returned BLAKE3 digests. Original crop pixels, not fallible OCR text alone, are authoritative for any source correction. Each strict entry names one element plus nonempty language-tagged source and/or target translation; all targets must belong to the page. Geometry, source-region ownership, opacity, padding, typography, and layout fields are not accepted. All four evidence artifacts' page/revision/digest provenance and first OCR are retained. Before initial review_pages, the four-part set authorizes a semantic batch. Once review_pages produces a deterministic plan, its first failure remains binding: only its allowed semantic field may be edited, and any layout/clearance plan blocks this tool. Accepted or failed visual/semantic review is never repair evidence.",
            ),
            definition::<PreviewTextLayout>(
                "preview_text_layout",
                "Rasterize a controlled, non-mutating target-fit candidate for one translated element. Ordinary options are constrained Korean line breaks, max lines, non-justify alignment, font scale decrease only, and positive safe-padding increase; raw bounds and replacement text are impossible. It may migrate only to an already-bound verified UI-panel anchor or a detector-owned adjacent free-dialogue anchor whose required compact no-container role, Japanese-vertical/Korean-horizontal modes, source-scaled proximity, original-pixel text/structure/boundary/background/contrast evidence, target-length room, reading order, and attribution all passed. Candidate discovery alone never changes the authoritative relation or placement. When every adjacent patch failed, the separately recorded source-bound fallback options object must be exactly {\"source_bound_interjection_fallback\":\"native_vertical\"}; the host supplies its native line, centering, font, and inset defaults. Preview must preserve semantic ownership and finite positive visible rendering while resolving the blocking page-level safety failure; font, glyph, density, contour-clearance, and anchor measurements remain diagnostic.",
            ),
            definition::<CommitTextLayout>(
                "commit_text_layout",
                "Commit only a current preview_text_layout preview_id that still preserves semantic identity, finite positive visible rendering, page containment, translated-text overlap safety, and strict improvement of the blocking deterministic result. Font, glyph, line-count, contour-clearance, density, and target-anchor measurements remain diagnostic. A verified target becomes authoritative only through this successful exact preview commit; source crop/OCR identity remains unchanged. Unknown, stale, unsafe, or non-improving previews do not mutate the project.",
            ),
            definition::<PreviewCompactTranslation>(
                "preview_compact_translation",
                "Submit one narrowly authorized Korean wording compaction only when the active deterministic repair plan names this tool for its exact logical dialogue group and primary render element. Requires the current four evidence digests and a nonempty ko-KR candidate with strictly fewer visible grapheme units. Source/OCR, member order, typography, geometry, and layout are immutable. The host previews it, revalidates semantic ownership, finite positive visible rendering, page containment, translated-text overlap safety, and strict improvement, commits the exact passing preview, and immediately reruns deterministic review before returning control.",
            ),
            definition::<CommitCompactTranslation>(
                "commit_compact_translation",
                "Commit only a valid current preview_compact_translation preview id. The host revalidates revision, exact active group/primary/member order, all four evidence bindings, finite positive visible rendering, page containment, and translated-text overlap safety before changing only the target translation. Unknown, failed, or stale previews do not mutate. The committed wording must then receive explicit in-loop meaning-fidelity and Korean-naturalness review against every original member crop before export.",
            ),
            definition::<PreviewTextSafeLayoutRepair>(
                "preview_increase_text_safe_padding",
                "Preview the only permitted deterministic text-safe-containment repair: increase every edge of the current text layout's safe inset by a finite positive delta. The host constructs geometry without accepting raw bounds, rasterizes the candidate without mutating the project, rejects geometry expansion or non-improving clearance, and returns measured before/candidate clearance plus the exact commit next_action. A preview is accepted only when minimum clearance strictly increases and reaches the unchanged configured threshold.",
            ),
            definition::<CommitTextSafeLayoutRepair>(
                "commit_text_safe_layout_repair",
                "Commit an accepted text-safe layout preview by its preview_id. The host rejects stale or unknown previews and remeasures the unchanged current/candidate revisions before committing. If clearance does not still strictly increase and satisfy the configured threshold, no mutation is made.",
            ),
            definition::<ExportPages>(
                "export_pages",
                "Export every imported page through Koharu's native exporter to the runner-configured required output directory and format. Deterministic language/layout acceptance must pass. Visual/semantic review must also pass through the configured command judge when present, or through fresh page-bound submit_visual_semantic_review decisions for every page otherwise.",
            ),
        ]
    })
}

fn tool_definition(name: &str) -> &'static Tool {
    tool_definitions()
        .iter()
        .find(|tool| tool.name == name)
        .unwrap_or_else(|| panic!("missing Koharu harness tool definition {name}"))
}

struct SessionCommitter {
    session: Arc<Mutex<Session>>,
}

#[async_trait]
impl Committer for SessionCommitter {
    async fn commit(&mut self, output: StageOutput) -> Result<Snapshot> {
        Ok(self
            .session
            .lock()
            .await
            .commit(output.patch)
            .await?
            .snapshot)
    }
}

fn definition<T: JsonSchema>(name: &str, description: &str) -> Tool {
    let _ = description;
    Tool::new(
        name,
        compact_tool_description(name),
        serde_json::to_value(schema_for!(T)).expect("tool argument schema must serialize"),
    )
}

#[cfg(test)]
const NATIVE_VERTICAL_FALLBACK_PAYLOAD_EXAMPLE: &str = r#"{"element":"<ELEMENT_ID>","options":{"source_bound_interjection_fallback":"native_vertical"},"reason":"use recorded native fallback"}"#;
const PREVIEW_TEXT_LAYOUT_DESCRIPTION: &str = "Preview the active deterministic target-fit operation. The live schema exposes only that operation's controlled options; native fallback appears only for recorded no-adjacent-candidate-passed evidence. All evidence, geometry, and rerender gates remain.";

fn compact_tool_description(name: &str) -> &'static str {
    match name {
        "inspect_project" => {
            "Read the authorized compact diagnostic project index and its complete artifact reference."
        }
        "view_page" => {
            "Render one page selected by its 1-based position in project.pages for read-only visual inspection."
        }
        "run_source_analysis" => {
            "Run detection and OCR only, then host-bind exact source-contained UI panel anchors; no translation or inpainting."
        }
        "inspect_source_evidence" => {
            "Write the authoritative original-page dossier and ordinal-linked crops for the page selected by its 1-based position in project.pages. Original pixels outrank OCR."
        }
        "view_page_source_debug" => {
            "Render the read-only original-page overlay for the page selected by its 1-based position in project.pages; does not mutate project state."
        }
        "classify_decorative_sfx" => {
            "On the page selected by its 1-based position in project.pages, resolve exact fresh-evidence uncontained dialogue/free-text screening: translate or skip only positively evidenced decorative SFX; retain_required preserves non-SFX required text on its ordinary path."
        }
        "verify_ui_panel_anchor" => {
            "On the page selected by its 1-based position in project.pages, bind required UI text to one listed detector-backed finite panel using exact fresh source evidence, containment, UI role, safe interior, provenance, confidence >= 0.90, and reason; inferred/nearest geometry is forbidden."
        }
        "inspect_page_evidence" => {
            "For the page selected by its 1-based position in project.pages, return the bundled source dossier/debug and translated dossier/debug artifacts with four exact current-revision digests."
        }
        "run_pipeline" => {
            "Translate and inpaint every required item after source review; only recorded difficult SFX are excluded."
        }
        "review_pages" => {
            "Create fresh original/render/semantic artifacts, run deterministic acceptance, and emit the binding first repair plan; a configured judge remains authoritative."
        }
        "submit_visual_semantic_review" => {
            "For the page selected by its 1-based position in project.pages, submit the seven page judgments with exact current revision and four artifact digests. Deterministic geometry cannot prove semantics; every skip and compacted group must be checked."
        }
        "revise_element" => {
            "Apply only fields authorized for the binding reviewed failure; skipped SFX and text-safe failures are not eligible."
        }
        "revise_page_translation" => {
            "For the page selected by its 1-based position in project.pages, revise source/target semantics using all four fresh page artifacts. Original crops are authoritative; geometry and typography are forbidden."
        }
        "preview_text_layout" => PREVIEW_TEXT_LAYOUT_DESCRIPTION,
        "commit_text_layout" => {
            "Commit only the current safe, strictly improving host preview after every deterministic gate is revalidated."
        }
        "preview_compact_translation" => {
            "Preview only the plan-authorized exact logical group with fresh four-part evidence and strictly shorter ko-KR text; source, order, typography, and geometry stay immutable."
        }
        "commit_compact_translation" => {
            "Commit only the current host preview after revision, group membership, evidence, and deterministic gates are revalidated."
        }
        "preview_increase_text_safe_padding" => {
            "Preview only positive per-edge safe-inset growth for the plan's exact element; raw bounds and geometry expansion are forbidden and applicable hard gates must pass."
        }
        "commit_text_safe_layout_repair" => {
            "Commit only the current host preview after clearance improvement and all applicable hard gates are revalidated."
        }
        "export_pages" => {
            "Export every page to the runner-configured destination and format only after current deterministic acceptance and required semantic review pass."
        }
        _ => "Koharu harness tool.",
    }
}

fn arguments<T: for<'de> Deserialize<'de>>(call: &ToolCall) -> Result<T> {
    serde_json::from_str(&call.arguments)
        .with_context(|| format!("invalid arguments for {}", call.name))
}

fn resolve_page_ordinal(snapshot: &Snapshot, page_ordinal: PageOrdinal) -> Result<PageTarget> {
    let page_count = snapshot.pages().count();
    let id = snapshot
        .pages()
        .nth(page_ordinal.get() - 1)
        .map(|page| page.id())
        .with_context(|| {
            format!(
                "page ordinal {} is out of range for project with {page_count} pages",
                page_ordinal.get()
            )
        })?;
    Ok(PageTarget {
        ordinal: page_ordinal,
        id,
    })
}

fn entity(value: &str) -> Result<EntityId> {
    serde_json::from_value(Value::String(value.to_owned()))
        .with_context(|| format!("invalid entity ID {value}"))
}

struct PreparedDecorativeSfxClassification {
    element_id: EntityId,
    content_id: EntityId,
    source_region_id: EntityId,
    source_ocr: SemanticText,
    source_typography: Typography,
    original_ordinal: usize,
    request: DecorativeSfxClassification,
}

fn decorative_sfx_disposition_candidates(inspection: &ProjectInspection) -> BTreeSet<EntityId> {
    inspection
        .project
        .pages
        .iter()
        .flat_map(|page| &page.text_elements)
        .filter(|element| requires_decorative_sfx_disposition(element))
        .map(|element| element.id)
        .collect()
}

fn screening_obligation(
    snapshot: &Snapshot,
    pending: &BTreeSet<EntityId>,
    evidence: &PageSemanticEvidenceState,
) -> Result<ScreeningObligation> {
    for (page_index, page) in snapshot.pages().enumerate() {
        let Some(group) = page.text_group()? else {
            continue;
        };
        let mut ordered = Vec::new();
        for layer in group.text_layers()? {
            let content = layer.content()?;
            let source_region = content.source_region()?;
            let detected = source_region.is_some_and(|region| {
                region
                    .region()
                    .and_then(|value| {
                        Ok(value.kind == TextRegion::kind() && region.detection()?.is_some())
                    })
                    .unwrap_or(false)
            });
            if !detected && content.source()?.is_none() && content.translation()?.is_none() {
                continue;
            }
            let bounds = source_region
                .map(|region| region.geometry())
                .transpose()?
                .or_else(|| {
                    layer
                        .balloon_target()
                        .ok()
                        .flatten()
                        .and_then(|region| region.geometry().ok())
                })
                .or_else(|| {
                    layer
                        .fit_target()
                        .ok()
                        .flatten()
                        .and_then(|region| region.geometry().ok())
                })
                .or(snapshot.component::<Geometry>(layer.id())?)
                .map(element_geometry)
                .map(|geometry| geometry.bounds);
            ordered.push((layer.id(), bounds));
        }
        ordered.sort_by(|(first_id, first_bounds), (second_id, second_bounds)| {
            match (first_bounds, second_bounds) {
                (Some(first), Some(second)) => first
                    .y
                    .total_cmp(&second.y)
                    .then_with(|| second.x.total_cmp(&first.x))
                    .then_with(|| first_id.to_string().cmp(&second_id.to_string())),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => first_id.to_string().cmp(&second_id.to_string()),
            }
        });
        let candidates = ordered
            .iter()
            .enumerate()
            .filter_map(|(index, (element_id, _))| {
                pending
                    .contains(element_id)
                    .then_some(ScreeningCandidateTarget {
                        element_id: *element_id,
                        original_ordinal: index + 1,
                    })
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            continue;
        }
        let page = PageTarget {
            ordinal: PageOrdinal(
                NonZeroUsize::new(page_index + 1).expect("page traversal is one-based"),
            ),
            id: page.id(),
        };
        let page_evidence = evidence.page(page.id);
        let source_dossier_is_current = page_evidence
            .and_then(|page| page.source_dossier.as_ref())
            .is_some_and(|artifact| {
                artifact.revision == snapshot.revision()
                    && artifact.page_id == page.id
                    && candidates
                        .iter()
                        .all(|candidate| artifact.element_crops.contains_key(&candidate.element_id))
            });
        let source_debug_is_current = page_evidence
            .and_then(|page| page.source_debug_artifact.as_ref())
            .is_some_and(|artifact| {
                artifact.revision == snapshot.revision() && artifact.page_id == page.id
            });
        let required_evidence_stage = if !source_dossier_is_current {
            ScreeningEvidenceStage::InspectSourceEvidence
        } else if !source_debug_is_current {
            ScreeningEvidenceStage::ViewPageSourceDebug
        } else {
            ScreeningEvidenceStage::ClassifyDecorativeSfx
        };
        return Ok(ScreeningObligation {
            page,
            candidates,
            required_evidence_stage,
        });
    }
    bail!(
        "{} pending decorative-SFX screening candidate(s) do not resolve in the current project traversal",
        pending.len()
    )
}

fn semantic_review_obligation(
    snapshot: &Snapshot,
    review: &VisualReviewRecord,
    page_reviews: &PageReviewState,
    evidence: &PageSemanticEvidenceState,
) -> Result<Option<SemanticReviewObligation>> {
    let bundled_pages = review
        .bundle
        .pages
        .iter()
        .map(|page| page.page_id)
        .collect::<BTreeSet<_>>();
    let mut visited = BTreeSet::new();
    for (page_index, page) in snapshot.pages().enumerate() {
        let page_id = page.id();
        if !bundled_pages.contains(&page_id) {
            continue;
        }
        visited.insert(page_id);
        if page_reviews
            .current_review(page_id)
            .is_some_and(|submitted| submitted.decision.accepted)
        {
            continue;
        }
        let target = PageTarget {
            ordinal: PageOrdinal(
                NonZeroUsize::new(page_index + 1).expect("page traversal is one-based"),
            ),
            id: page_id,
        };
        let required_stage =
            if complete_page_evidence(evidence, page_id).is_some_and(|(revision, evidence_page)| {
                revision == snapshot.revision() && evidence_page == page_id
            }) {
                SemanticReviewStage::SubmitVisualSemanticReview
            } else {
                SemanticReviewStage::InspectPageEvidence
            };
        return Ok(Some(SemanticReviewObligation {
            page: target,
            required_stage,
        }));
    }
    ensure!(
        visited == bundled_pages,
        "visual-review bundle pages do not match the current project page traversal"
    );
    Ok(None)
}

fn requires_decorative_sfx_disposition(element: &TextElementInspection) -> bool {
    element.detected
        && element.required
        && element.source_region_id.is_some()
        && element
            .source_geometry
            .as_ref()
            .is_some_and(valid_element_geometry)
        && matches!(
            element.text_role.as_deref(),
            Some(FREE_TEXT_ROLE | crate::free_dialogue::DIALOGUE_ROLE)
        )
        && element.decorative_sfx.is_none()
        && element.verified_ui_panel_anchor.is_none()
        && !is_actual_container_bound(element)
}

fn validate_decorative_sfx_classifications(
    page: &PageInspection,
    observations: &PageSemanticEvidenceState,
    revision: Revision,
    arguments: &ClassifyDecorativeSfx,
) -> Result<Vec<PreparedDecorativeSfxClassification>> {
    if arguments.decisions.is_empty() {
        bail!("classify_decorative_sfx requires at least one classification decision");
    }
    let observations = observations
        .page(page.id)
        .context("classify_decorative_sfx has no evidence for the requested page")?;
    let source_dossier = observations.source_dossier.as_ref().context(
        "classify_decorative_sfx requires inspect_source_evidence for the current revision",
    )?;
    let source_debug = observations.source_debug_artifact.as_ref().context(
        "classify_decorative_sfx requires view_page_source_debug for the current revision",
    )?;
    for (name, evidence) in [
        ("source dossier", source_dossier),
        ("source debug artifact", source_debug),
    ] {
        if evidence.revision != revision || evidence.page_id != page.id {
            bail!("classify_decorative_sfx {name} evidence is stale or belongs to another page");
        }
    }
    if source_dossier.blake3 != arguments.source_evidence_dossier_blake3 {
        bail!("classify_decorative_sfx source evidence dossier digest does not match");
    }
    if source_debug.blake3 != arguments.source_debug_artifact_blake3 {
        bail!("classify_decorative_sfx source debug artifact digest does not match");
    }

    let mut unique = std::collections::BTreeSet::new();
    let mut prepared = Vec::with_capacity(arguments.decisions.len());
    for request in &arguments.decisions {
        let element_id = entity(&request.element)?;
        if !unique.insert(element_id) {
            bail!("decorative-SFX classification repeats element {element_id}");
        }
        let element = page
            .text_elements
            .iter()
            .find(|element| element.id == element_id)
            .with_context(|| {
                format!("classified element {element_id} is not on page {}", page.id)
            })?;
        if !requires_decorative_sfx_disposition(element) {
            bail!(
                "element {element_id} is not a required detector-backed, valid-geometry, uncontained dialogue/free-text screening candidate"
            );
        }
        let source_ocr = element.source.clone().with_context(|| {
            format!("element {element_id} has no OCR source record to preserve")
        })?;
        let source_typography = element.typography.clone().with_context(|| {
            format!("element {element_id} has no detected typography to preserve")
        })?;
        let source_evidence = source_dossier
            .element_crops
            .get(&element_id)
            .with_context(|| {
                format!("element {element_id} has no authoritative source crop evidence")
            })?;
        if request.original_ordinal == 0
            || request.original_ordinal != source_evidence.ordinal
            || request.source_crop_blake3 != source_evidence.crop_blake3
            || request.source_debug_label != source_evidence.source_debug_label
        {
            bail!(
                "element {element_id} ordinal, crop digest, or source-debug label does not match current source evidence"
            );
        }
        if !request.confidence.is_finite()
            || !(MINIMUM_SFX_CLASSIFICATION_CONFIDENCE..=1.0).contains(&request.confidence)
        {
            bail!(
                "element {element_id} screening confidence must be between {MINIMUM_SFX_CLASSIFICATION_CONFIDENCE:.2} and 1.0"
            );
        }
        let evidence_fields = [
            request.evidence.decorative_visual_form.trim(),
            request.evidence.sound_effect_page_function.trim(),
            request.evidence.legibility_and_translation_value.trim(),
            request
                .evidence
                .exclusion_of_dialogue_caption_ui_and_general_free_text
                .trim(),
        ];
        if evidence_fields.iter().any(|value| value.is_empty())
            || evidence_fields
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != evidence_fields.len()
            || request.rationale.trim().is_empty()
        {
            bail!(
                "element {element_id} requires distinct visual-form, page-function, legibility/translation-value, required-content-role evidence, and a rationale"
            );
        }
        prepared.push(PreparedDecorativeSfxClassification {
            element_id,
            content_id: element.content_id,
            source_region_id: element.source_region_id.expect("validated source region"),
            source_ocr,
            source_typography,
            original_ordinal: request.original_ordinal,
            request: request.clone(),
        });
    }
    Ok(prepared)
}

struct PreparedUiPanelVerification {
    element_id: EntityId,
    content_id: EntityId,
    source_region_id: EntityId,
    source_ocr: SemanticText,
    original_ordinal: usize,
    panel: DetectedPanelCandidate,
    source_containment_ratio: f64,
    source_intersection_ratio: f64,
    request: UiPanelVerification,
}

fn prepare_deterministic_ui_panel_bindings(
    page: &PageInspection,
    source_evidence: &BTreeMap<EntityId, SourceElementEvidence>,
    evidence_revision: Revision,
    decision_revision: Revision,
) -> Result<Vec<UiPanelAnchorDecision>> {
    const HOST_UI_ROLE_EVIDENCE: &str =
        "exact required UI text role from the source-raster closed-panel candidate";
    let mut prepared = Vec::new();
    for panel in &page.detected_panel_candidates {
        let source_region_id = panel.raster_evidence.source_relationship.source_region_id;
        if page
            .detected_panel_candidates
            .iter()
            .filter(|candidate| {
                candidate
                    .raster_evidence
                    .source_relationship
                    .source_region_id
                    == source_region_id
            })
            .count()
            != 1
        {
            continue;
        }
        let matching = page
            .text_elements
            .iter()
            .filter(|element| element.source_region_id == Some(source_region_id))
            .collect::<Vec<_>>();
        let [element] = matching.as_slice() else {
            continue;
        };
        let Some(source_geometry) = element.source_geometry.as_ref() else {
            continue;
        };
        let source_bound = !is_actual_container_bound(element);
        if !element.detected
            || !element.required
            || element.decorative_sfx.is_some()
            || element.text_role.as_deref() != Some(FREE_TEXT_ROLE)
            || !element.logical_dialogue_memberships.is_empty()
            || element.verified_ui_panel_anchor.is_some()
            || !source_bound
            || panel.detection_label != "source-raster-closed-ui-panel"
            || panel.region_kind != PanelRegion::KIND
            || !valid_element_geometry(&panel.geometry)
            || !panel.detection_confidence.is_finite()
            || panel.detection_confidence < MINIMUM_UI_PANEL_VERIFICATION_CONFIDENCE
            || !valid_bounds(panel.raster_evidence.safe_interior_bbox)
            || panel.raster_evidence.panel_bbox != panel.geometry.bounds
            || panel.detector_producer != panel.raster_evidence.detector.producer
            || panel.detector_model.as_deref() != Some(panel.raster_evidence.detector.version)
        {
            continue;
        }
        raster_candidate_supports_ui_role(
            &panel.raster_evidence,
            source_region_id,
            source_geometry.bounds,
            HOST_UI_ROLE_EVIDENCE,
        )
        .map_err(|reason| {
            anyhow!(
                "deterministic panel {} cannot bind exact source {}: {reason}",
                panel.region_id,
                source_region_id
            )
        })?;
        let safe_interior = panel.raster_evidence.safe_interior_bbox;
        let source_containment_ratio =
            bounds_intersection_area(source_geometry.bounds, safe_interior)
                / (source_geometry.bounds.width * source_geometry.bounds.height);
        let source_intersection_ratio =
            bounds_intersection_area(source_geometry.bounds, panel.geometry.bounds)
                / (panel.geometry.bounds.width * panel.geometry.bounds.height);
        if !source_containment_ratio.is_finite()
            || source_containment_ratio < 0.90
            || !source_intersection_ratio.is_finite()
            || source_intersection_ratio <= 0.0
        {
            continue;
        }
        let Some(source_ocr) = element.source.clone() else {
            continue;
        };
        let Some(source_evidence) = source_evidence.get(&element.id) else {
            continue;
        };
        prepared.push(UiPanelAnchorDecision {
            schema_version: 3,
            decision: "verified_required_ui_panel_text_anchor",
            evidence_revision,
            decision_revision,
            page_id: page.id,
            original_ordinal: source_evidence.ordinal,
            element_id: element.id,
            content_id: element.content_id,
            source_region_id,
            source_ocr,
            source_crop_blake3: source_evidence.crop_blake3.clone(),
            source_debug_label: source_evidence.source_debug_label.clone(),
            panel: panel.clone(),
            source_containment_ratio,
            source_intersection_ratio,
            classifier: UiPanelClassifier {
                kind: "host_deterministic_source_raster",
                configured_model: None,
                tool_call_id: "host:source-analysis:ui-panel-binding".to_owned(),
            },
            evidence: UiPanelEvidence {
                ui_role_and_function: HOST_UI_ROLE_EVIDENCE.to_owned(),
                finite_visible_panel_or_screen:
                    "finite positive closed panel geometry measured from original source pixels"
                        .to_owned(),
                source_to_panel_relation:
                    "exact source region is contained by this detector-owned panel safe interior"
                        .to_owned(),
                text_safe_interior:
                    "finite detector-measured safe interior with minimum font and clearance room"
                        .to_owned(),
                visual_or_detection_provenance:
                    "closed-rectangle-v1 detector over unchanged original source pixels".to_owned(),
            },
            confidence: panel.raster_evidence.confidence,
            association_reason:
                "host_exact_source_contained_in_detector_backed_ui_panel_safe_interior".to_owned(),
        });
    }
    Ok(prepared)
}

fn validate_ui_panel_verifications(
    page: &PageInspection,
    observations: &PageSemanticEvidenceState,
    revision: Revision,
    arguments: &VerifyUiPanelAnchor,
) -> Result<Vec<PreparedUiPanelVerification>> {
    if arguments.decisions.is_empty() {
        bail!("verify_ui_panel_anchor requires at least one positive verification");
    }
    let observations = observations
        .page(page.id)
        .context("verify_ui_panel_anchor has no evidence for the requested page")?;
    let source_dossier = observations.source_dossier.as_ref().context(
        "verify_ui_panel_anchor requires inspect_source_evidence for the current revision",
    )?;
    let source_debug = observations.source_debug_artifact.as_ref().context(
        "verify_ui_panel_anchor requires view_page_source_debug for the current revision",
    )?;
    for (name, evidence) in [
        ("source dossier", source_dossier),
        ("source debug artifact", source_debug),
    ] {
        if evidence.revision != revision || evidence.page_id != page.id {
            bail!("verify_ui_panel_anchor {name} evidence is stale or belongs to another page");
        }
    }
    if source_dossier.blake3 != arguments.source_evidence_dossier_blake3 {
        bail!("verify_ui_panel_anchor source evidence dossier digest does not match");
    }
    if source_debug.blake3 != arguments.source_debug_artifact_blake3 {
        bail!("verify_ui_panel_anchor source debug artifact digest does not match");
    }

    let mut unique_elements = std::collections::BTreeSet::new();
    let mut unique_panels = std::collections::BTreeSet::new();
    let mut prepared = Vec::with_capacity(arguments.decisions.len());
    for request in &arguments.decisions {
        let element_id = entity(&request.element)?;
        if !unique_elements.insert(element_id) {
            bail!("UI-panel verification repeats element {element_id}");
        }
        let panel_id = entity(&request.panel_region)?;
        if !unique_panels.insert((element_id, panel_id)) {
            bail!("UI-panel verification repeats element/panel association");
        }
        let element = page
            .text_elements
            .iter()
            .find(|element| element.id == element_id)
            .with_context(|| format!("verified element {element_id} is not on page {}", page.id))?;
        if !element.detected
            || !element.required
            || element.source_region_id.is_none()
            || element.source_geometry.is_none()
        {
            bail!("element {element_id} is not required detector-backed source text");
        }
        if element.decorative_sfx.is_some()
            || element.text_role.as_deref() != Some(FREE_TEXT_ROLE)
            || !element.logical_dialogue_memberships.is_empty()
            || is_actual_container_bound(element)
            || element.verified_ui_panel_anchor.is_some()
        {
            bail!(
                "element {element_id} is SFX, dialogue, caption/UI, generic non-free text, already verified, or container-associated; it cannot claim a UI-panel anchor"
            );
        }
        let panel = page
            .detected_panel_candidates
            .iter()
            .find(|panel| panel.region_id == panel_id)
            .with_context(|| {
                format!(
                    "panel {panel_id} is not an explicit detector-backed panel candidate on page {}",
                    page.id
                )
            })?;
        if panel.region_kind != PanelRegion::KIND
            || panel.region_id == element.source_region_id.expect("validated source region")
            || !valid_element_geometry(&panel.geometry)
            || !panel.detection_confidence.is_finite()
            || panel.detector_producer.trim().is_empty()
        {
            bail!("panel {panel_id} lacks finite robust panel-detection provenance");
        }
        let source = element
            .source_geometry
            .as_ref()
            .expect("validated geometry");
        let raster_evidence = &panel.raster_evidence;
        if panel.detector_producer != raster_evidence.detector.producer
            || panel.detector_model.as_deref() != Some(raster_evidence.detector.version)
        {
            bail!("panel {panel_id} raster and scene detector provenance disagree");
        }
        raster_candidate_supports_ui_role(
            raster_evidence,
            element.source_region_id.expect("validated source region"),
            source.bounds,
            &request.evidence.ui_role_and_function,
        )
        .map_err(|reason| {
            anyhow!("panel {panel_id} cannot bind element {element_id} as UI: {reason}")
        })?;
        if raster_evidence.panel_bbox != panel.geometry.bounds {
            bail!("panel {panel_id} raster evidence no longer matches its scene geometry");
        }
        let intersection = bounds_intersection_area(source.bounds, panel.geometry.bounds);
        let source_area = source.bounds.width * source.bounds.height;
        let panel_area = panel.geometry.bounds.width * panel.geometry.bounds.height;
        let source_containment_ratio = intersection / source_area;
        let source_intersection_ratio = intersection / panel_area;
        if !source_containment_ratio.is_finite()
            || source_containment_ratio < 0.90
            || !source_intersection_ratio.is_finite()
            || source_intersection_ratio <= 0.0
        {
            bail!(
                "element {element_id} source geometry is not reliably contained by explicit panel {panel_id}; nearest-region or weak-intersection association is forbidden"
            );
        }
        let source_ocr = element
            .source
            .clone()
            .with_context(|| format!("element {element_id} has no OCR source record to retain"))?;
        let source_evidence = source_dossier
            .element_crops
            .get(&element_id)
            .with_context(|| format!("element {element_id} has no authoritative source crop"))?;
        if request.original_ordinal == 0
            || request.original_ordinal != source_evidence.ordinal
            || request.source_crop_blake3 != source_evidence.crop_blake3
            || request.source_debug_label != source_evidence.source_debug_label
        {
            bail!(
                "element {element_id} ordinal, crop digest, or source-debug label does not match current source evidence"
            );
        }
        if !request.confidence.is_finite()
            || !(MINIMUM_UI_PANEL_VERIFICATION_CONFIDENCE..=1.0).contains(&request.confidence)
        {
            bail!(
                "element {element_id} UI-panel verification confidence must be between {MINIMUM_UI_PANEL_VERIFICATION_CONFIDENCE:.2} and 1.0"
            );
        }
        let evidence_fields = [
            request.evidence.ui_role_and_function.trim(),
            request.evidence.finite_visible_panel_or_screen.trim(),
            request.evidence.source_to_panel_relation.trim(),
            request.evidence.text_safe_interior.trim(),
            request.evidence.visual_or_detection_provenance.trim(),
        ];
        if evidence_fields.iter().any(|value| value.is_empty())
            || evidence_fields
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != evidence_fields.len()
            || request.association_reason.trim().is_empty()
        {
            bail!(
                "element {element_id} requires distinct UI-role, finite-panel, source-relation, text-safe-interior, provenance evidence, and an association reason"
            );
        }
        match request.classification {
            UiPanelClassificationKind::RequiredUiPanelText => {}
        }
        prepared.push(PreparedUiPanelVerification {
            element_id,
            content_id: element.content_id,
            source_region_id: element.source_region_id.expect("validated source region"),
            source_ocr,
            original_ordinal: request.original_ordinal,
            panel: panel.clone(),
            source_containment_ratio,
            source_intersection_ratio,
            request: request.clone(),
        });
    }
    Ok(prepared)
}

fn valid_element_geometry(geometry: &ElementGeometry) -> bool {
    geometry.points.len() >= 3
        && [
            geometry.bounds.x,
            geometry.bounds.y,
            geometry.bounds.width,
            geometry.bounds.height,
        ]
        .into_iter()
        .all(f64::is_finite)
        && geometry.bounds.width > 0.0
        && geometry.bounds.height > 0.0
        && geometry
            .points
            .iter()
            .all(|point| point.x.is_finite() && point.y.is_finite())
}

fn bounds_intersection_area(first: ElementBounds, second: ElementBounds) -> f64 {
    let width = (first.x + first.width).min(second.x + second.width) - first.x.max(second.x);
    let height = (first.y + first.height).min(second.y + second.height) - first.y.max(second.y);
    width.max(0.0) * height.max(0.0)
}

fn page_containing(snapshot: &Snapshot, entity: EntityId) -> Result<EntityId> {
    snapshot.entity(entity)?;
    let mut current = entity;
    loop {
        if snapshot.page(current).is_ok() {
            return Ok(current);
        }
        current = snapshot
            .parent(current)?
            .with_context(|| format!("entity {entity} is not contained by a page"))?;
    }
}

fn repair_evidence(
    review: &VisualReviewRecord,
    observation: Option<&RevisionEvidence>,
    current_revision: koharu_scene::Revision,
    element_page: EntityId,
) -> Result<RevisionEvidence> {
    if review.scene_revision != current_revision {
        bail!("review_pages must evaluate the current revision {current_revision} before repair");
    }
    if review.rejected() {
        return Ok(RevisionEvidence::ExternalReviewRejection {
            revision: review.scene_revision,
            review_attempt: review.attempt,
        });
    }
    if !review.deterministic_acceptance_passed {
        return Ok(RevisionEvidence::DeterministicRejection {
            revision: review.scene_revision,
            review_attempt: review.attempt,
        });
    }
    if review.status != VisualReviewStatus::PendingAgentReview {
        bail!(
            "repair requires deterministic rejection, visual/semantic review rejection, or pending agent review with explicit current-revision evidence"
        );
    }

    let observation = observation.context(
        "pending agent review requires inspect_project or view_page evidence from the current revision before repair",
    )?;
    if observation.revision() != current_revision {
        bail!(
            "pending agent review requires fresh inspect_project or view_page evidence from current revision {current_revision}"
        );
    }
    match observation {
        RevisionEvidence::ProjectInspection { .. } => Ok(observation.clone()),
        RevisionEvidence::RenderedPageInspection { page_id, .. }
        | RevisionEvidence::PageTranslationReview { page_id, .. }
        | RevisionEvidence::RenderedPageDebugInspection { page_id, .. }
        | RevisionEvidence::SourceEvidenceInspection { page_id, .. }
        | RevisionEvidence::SourcePageDebugInspection { page_id, .. }
        | RevisionEvidence::PageTranslationVisualEvidence { page_id, .. }
            if *page_id == element_page =>
        {
            Ok(observation.clone())
        }
        RevisionEvidence::RenderedPageInspection { .. }
        | RevisionEvidence::PageTranslationReview { .. }
        | RevisionEvidence::RenderedPageDebugInspection { .. }
        | RevisionEvidence::SourceEvidenceInspection { .. }
        | RevisionEvidence::SourcePageDebugInspection { .. }
        | RevisionEvidence::PageTranslationVisualEvidence { .. } => {
            bail!("page evidence must show the page containing the revised element")
        }
        RevisionEvidence::DeterministicRejection { .. }
        | RevisionEvidence::ExternalReviewRejection { .. } => {
            bail!("pending agent review requires explicit inspect_project or view_page evidence")
        }
    }
}

fn page_translation_evidence(
    review: Option<&VisualReviewRecord>,
    observations: &PageSemanticEvidenceState,
    current_scene_revision: Revision,
    evidence_revision: Revision,
    page: EntityId,
    source_evidence_dossier_blake3: &str,
    source_debug_artifact_blake3: &str,
    dossier_blake3: &str,
    debug_artifact_blake3: &str,
) -> Result<RevisionEvidence> {
    if let Some(review) = review {
        if review.scene_revision != current_scene_revision {
            bail!(
                "review_pages must evaluate the current revision {current_scene_revision} before another repair"
            );
        }
        if review.status == VisualReviewStatus::Accepted {
            bail!(
                "an accepted visual/semantic review cannot be used as page translation repair evidence"
            );
        }
        if review.status == VisualReviewStatus::Failed {
            bail!(
                "a failed visual/semantic review cannot be used as page translation repair evidence"
            );
        }
    }

    let evidence = validate_four_part_page_evidence(
        observations,
        evidence_revision,
        page,
        source_evidence_dossier_blake3,
        source_debug_artifact_blake3,
        dossier_blake3,
        debug_artifact_blake3,
        "revise_page_translation",
    )?;

    Ok(RevisionEvidence::PageTranslationVisualEvidence {
        revision: evidence_revision,
        page_id: page,
        source_evidence_dossier_blake3: evidence.source_evidence_dossier_blake3,
        source_debug_artifact_blake3: evidence.source_debug_artifact_blake3,
        dossier_blake3: evidence.dossier_blake3,
        debug_artifact_blake3: evidence.debug_artifact_blake3,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ValidatedPageEvidence {
    source_evidence_dossier_blake3: String,
    source_debug_artifact_blake3: String,
    dossier_blake3: String,
    debug_artifact_blake3: String,
}

#[allow(clippy::too_many_arguments)]
fn validate_four_part_page_evidence(
    observations: &PageSemanticEvidenceState,
    current_revision: Revision,
    page: EntityId,
    source_evidence_dossier_blake3: &str,
    source_debug_artifact_blake3: &str,
    dossier_blake3: &str,
    debug_artifact_blake3: &str,
    action: &str,
) -> Result<ValidatedPageEvidence> {
    if [
        source_evidence_dossier_blake3,
        source_debug_artifact_blake3,
        dossier_blake3,
        debug_artifact_blake3,
    ]
    .iter()
    .any(|digest| digest.trim().is_empty())
    {
        bail!("{action} requires all four nonempty source and translated evidence digests");
    }

    let observations = observations
        .page(page)
        .with_context(|| format!("{action} has no evidence for page {page}"))?;

    let source_dossier = observations
        .source_dossier
        .as_ref()
        .with_context(|| format!("{action} requires fresh bundled source dossier evidence"))?;
    let source_debug = observations
        .source_debug_artifact
        .as_ref()
        .with_context(|| format!("{action} requires fresh bundled source debug evidence"))?;
    let dossier = observations
        .dossier
        .as_ref()
        .with_context(|| format!("{action} requires fresh bundled translated dossier evidence"))?;
    let debug = observations
        .debug_artifact
        .as_ref()
        .with_context(|| format!("{action} requires fresh bundled rendered debug evidence"))?;
    for (name, evidence) in [
        ("source evidence dossier", source_dossier),
        ("source debug artifact", source_debug),
        ("translated dossier", dossier),
        ("translated debug artifact", debug),
    ] {
        if evidence.revision != current_revision {
            bail!(
                "{action} {name} evidence is stale: evidence revision {}, current revision {current_revision}",
                evidence.revision
            );
        }
        if evidence.page_id != page {
            bail!(
                "{action} {name} evidence is for page {}, not requested page {page}",
                evidence.page_id
            );
        }
    }
    if source_dossier.blake3 != source_evidence_dossier_blake3 {
        bail!(
            "{action} source evidence dossier digest does not match fresh source inspection evidence"
        );
    }
    if source_debug.blake3 != source_debug_artifact_blake3 {
        bail!(
            "{action} source debug artifact digest does not match fresh original-page visual evidence"
        );
    }
    if dossier.blake3 != dossier_blake3 {
        bail!("{action} dossier digest does not match fresh review evidence");
    }
    if debug.blake3 != debug_artifact_blake3 {
        bail!("{action} debug artifact digest does not match fresh visual evidence");
    }

    Ok(ValidatedPageEvidence {
        source_evidence_dossier_blake3: source_dossier.blake3.clone(),
        source_debug_artifact_blake3: source_debug.blake3.clone(),
        dossier_blake3: dossier.blake3.clone(),
        debug_artifact_blake3: debug.blake3.clone(),
    })
}

fn validate_agent_visual_semantic_review(
    review: &VisualReviewRecord,
    observations: &PageSemanticEvidenceState,
    evidence_revision: Revision,
    page: EntityId,
    compact_requirements: &[CompactTranslationReviewRequirement],
    arguments: &SubmitVisualSemanticReview,
) -> Result<AgentVisualSemanticReview> {
    if review.judge.required || review.judge.command.is_some() {
        bail!(
            "submit_visual_semantic_review cannot replace or bypass the configured command judge"
        );
    }
    if review.status != VisualReviewStatus::PendingAgentReview {
        bail!("submit_visual_semantic_review requires a pending in-loop review");
    }
    if Revision::new(arguments.scene_revision) != evidence_revision {
        bail!(
            "submit_visual_semantic_review revision binding does not match current page evidence revision {evidence_revision}"
        );
    }
    if !review
        .bundle
        .pages
        .iter()
        .any(|entry| entry.page_id == page)
    {
        bail!("submitted page {page} is not part of the current review bundle");
    }
    if review
        .agent_reviews
        .iter()
        .any(|submitted| submitted.page_id == page)
    {
        bail!("page {page} already has an in-loop review for the current revision");
    }
    arguments.decision.validate_agent_submission()?;
    let evidence = validate_four_part_page_evidence(
        observations,
        evidence_revision,
        page,
        &arguments.source_evidence_dossier_blake3,
        &arguments.source_debug_artifact_blake3,
        &arguments.dossier_blake3,
        &arguments.debug_artifact_blake3,
        "submit_visual_semantic_review",
    )?;
    validate_compacted_translation_semantic_reviews(
        observations,
        page,
        compact_requirements,
        &arguments.compacted_translation_reviews,
        &arguments.decision,
    )?;
    Ok(AgentVisualSemanticReview {
        schema_version: AGENT_VISUAL_SEMANTIC_REVIEW_SCHEMA_VERSION,
        page_id: page,
        scene_revision: evidence_revision,
        source_evidence_dossier_blake3: evidence.source_evidence_dossier_blake3,
        source_debug_artifact_blake3: evidence.source_debug_artifact_blake3,
        dossier_blake3: evidence.dossier_blake3,
        debug_artifact_blake3: evidence.debug_artifact_blake3,
        compacted_translation_reviews: arguments.compacted_translation_reviews.clone(),
        decision: arguments.decision.clone(),
    })
}

fn compact_translation_review_requirements(
    snapshot: &Snapshot,
    page: EntityId,
    corrections: &[CorrectionRecord],
) -> Result<Vec<CompactTranslationReviewRequirement>> {
    snapshot.page(page)?;
    let compacted_elements = corrections
        .iter()
        .filter(|record| record.agent_action.tool == "commit_compact_translation")
        .filter_map(|record| {
            (page_containing(snapshot, record.element_id).ok() == Some(page))
                .then_some(record.element_id)
        })
        .collect::<std::collections::BTreeSet<_>>();
    if compacted_elements.is_empty() {
        return Ok(Vec::new());
    }
    let mut requirements = Vec::new();
    let group = snapshot
        .page(page)?
        .text_group()?
        .context("compacted translation page has no text group")?;
    for layer in group.text_layers()? {
        let content = layer.content()?;
        let Some(dialogue) = content.logical_dialogue()? else {
            continue;
        };
        let primary = dialogue.members[0].element_id;
        if !compacted_elements.contains(&primary) {
            continue;
        }
        requirements.push(CompactTranslationReviewRequirement {
            logical_group_id: content.id(),
            primary_render_element_id: primary,
            members: dialogue
                .members
                .iter()
                .map(|member| crate::repair::RepairLogicalDialogueMember {
                    ordinal: member.ordinal,
                    element_id: member.element_id,
                    source_region_id: member.source_region_id,
                })
                .collect(),
        });
    }
    requirements.sort_by_key(|requirement| requirement.logical_group_id);
    if requirements.len() != compacted_elements.len() {
        bail!("a compacted translation no longer has its immutable logical dialogue group");
    }
    Ok(requirements)
}

fn validate_compacted_translation_semantic_reviews(
    observations: &PageSemanticEvidenceState,
    page: EntityId,
    requirements: &[CompactTranslationReviewRequirement],
    submitted: &[CompactedTranslationSemanticReview],
    decision: &VisualReviewDecision,
) -> Result<()> {
    if submitted.len() != requirements.len() {
        bail!(
            "submit_visual_semantic_review requires exactly {} compacted translation review(s), got {}",
            requirements.len(),
            submitted.len()
        );
    }
    let source_dossier = observations
        .page(page)
        .context("compacted translation review has no evidence for the requested page")?
        .source_dossier
        .as_ref()
        .context("compacted translation review requires fresh original member crops")?;
    let mut seen = std::collections::BTreeSet::new();
    for review in submitted {
        let logical_group = entity(&review.logical_group)?;
        let primary = entity(&review.primary_render_element)?;
        if !seen.insert((logical_group, primary)) {
            bail!("compacted translation semantic review contains a duplicate group");
        }
        let requirement = requirements
            .iter()
            .find(|required| {
                required.logical_group_id == logical_group
                    && required.primary_render_element_id == primary
            })
            .context("compacted translation semantic review targets an unexpected group")?;
        if review.rationale.trim().is_empty() {
            bail!("compacted translation semantic review rationale cannot be empty");
        }
        if review.issues.iter().any(|issue| issue.trim().is_empty()) {
            bail!("compacted translation semantic review issues cannot contain empty entries");
        }
        if review.member_evidence.len() != requirement.members.len() {
            bail!(
                "compacted translation semantic review must bind every ordered original group-member crop"
            );
        }
        for (submitted_member, required_member) in
            review.member_evidence.iter().zip(&requirement.members)
        {
            let submitted_element = entity(&submitted_member.element)?;
            let source_crop = source_dossier
                .element_crops
                .get(&required_member.element_id)
                .context("required original logical-group member crop is missing")?;
            if submitted_member.ordinal != required_member.ordinal
                || submitted_element != required_member.element_id
                || submitted_member.source_crop_blake3 != source_crop.crop_blake3
            {
                bail!(
                    "compacted translation semantic review member ordering or crop binding is stale"
                );
            }
        }
        let compact_passes = review.source_fidelity_preserved && review.target_language_natural;
        if compact_passes && !review.issues.is_empty() {
            bail!("accepted compacted translation review cannot contain issues");
        }
        if !compact_passes && review.issues.is_empty() {
            bail!("rejected compacted translation review must produce concrete issues");
        }
        if decision.accepted && !compact_passes {
            bail!("meaning drift or unnatural Korean in a compacted translation blocks acceptance");
        }
        if !review.source_fidelity_preserved && decision.judgments.translation_meaning_accurate {
            bail!("compacted meaning drift must reject translation_meaning_accurate");
        }
        if !review.target_language_natural && decision.judgments.target_language_natural {
            bail!("unnatural compacted Korean must reject target_language_natural");
        }
        for issue in &review.issues {
            if !decision.issues.contains(issue) {
                bail!("compacted translation rejection issues must be retained in the page review");
            }
        }
    }
    Ok(())
}

fn record_agent_visual_semantic_review(
    review: &mut VisualReviewRecord,
    submitted: AgentVisualSemanticReview,
) {
    review.agent_reviews.push(submitted);
    refresh_agent_visual_semantic_review_status(review);
}

fn refresh_agent_visual_semantic_review_status(review: &mut VisualReviewRecord) {
    review.status = if review
        .agent_reviews
        .iter()
        .any(|submitted| !submitted.decision.accepted)
    {
        VisualReviewStatus::Rejected
    } else if review.bundle.pages.iter().all(|page| {
        review
            .agent_reviews
            .iter()
            .any(|submitted| submitted.page_id == page.page_id && submitted.decision.accepted)
    }) {
        VisualReviewStatus::Accepted
    } else {
        VisualReviewStatus::PendingAgentReview
    };
}

fn validate_visual_review_export_precondition(
    review: &VisualReviewRecord,
    page_reviews: &PageReviewState,
    current_revision: Revision,
) -> Result<()> {
    if review.scene_revision != current_revision {
        bail!(
            "export rejected: visual/semantic review revision {} is stale for current revision {current_revision}",
            review.scene_revision
        );
    }
    if !review.accepted() {
        if review.judge.required {
            bail!("export rejected: configured command judge did not accept the current review");
        }
        bail!("export rejected: in-loop visual/semantic review did not accept every page");
    }
    if !review.judge.required {
        for page in &review.bundle.pages {
            let current = page_reviews.current_review(page.page_id).with_context(|| {
                format!(
                    "export rejected: page {} has no current visual/semantic review",
                    page.page_id
                )
            })?;
            let bundled = review
                .agent_reviews
                .iter()
                .find(|submitted| submitted.page_id == page.page_id)
                .context("export rejected: current page review is absent from the review bundle")?;
            if bundled.scene_revision != current.scene_revision
                || bundled.source_evidence_dossier_blake3 != current.source_evidence_dossier_blake3
                || bundled.source_debug_artifact_blake3 != current.source_debug_artifact_blake3
                || bundled.dossier_blake3 != current.dossier_blake3
                || bundled.debug_artifact_blake3 != current.debug_artifact_blake3
            {
                bail!(
                    "export rejected: bundled visual/semantic review for page {} is not the current page-bound decision",
                    page.page_id
                );
            }
        }
    }
    Ok(())
}

fn validate_page_translation_revision(
    arguments: &RevisePageTranslation,
    source_language: Language,
    target_language: Language,
) -> Result<()> {
    if arguments.page_rationale.trim().is_empty() {
        bail!("page-level translation rationale cannot be empty");
    }
    if arguments.edits.is_empty() {
        bail!("page translation revision requires at least one element edit");
    }
    for edit in &arguments.edits {
        if edit.source.is_none() && edit.translation.is_none() {
            bail!("each page translation entry must contain a source or translation edit");
        }
        for (name, text, expected_language) in [
            ("source", edit.source.as_ref(), source_language.tag()),
            (
                "translation",
                edit.translation.as_ref(),
                target_language.tag(),
            ),
        ] {
            if let Some(text) = text {
                if text.text.trim().is_empty() {
                    bail!("page translation {name} text cannot be empty");
                }
                if text.language != expected_language {
                    bail!(
                        "page translation {name} language must be {expected_language}, got {}",
                        text.language
                    );
                }
            }
        }
    }
    Ok(())
}

fn validate_page_semantic_plan(
    failure: Option<&DeterministicRepairFailure>,
    edits: &[(EntityId, &PageTranslationEdit)],
) -> Result<()> {
    let Some(failure) = failure else {
        return Ok(());
    };
    if failure.code == AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior
        || !failure.allowed_repair_fields.iter().any(|field| {
            matches!(
                field,
                RepairField::SourceText | RepairField::TranslationText
            )
        })
    {
        bail!(
            "the first active deterministic failure is a layout/clearance plan; revise_page_translation is blocked and its next_action must be used"
        );
    }
    let required = failure
        .primary_render_element_id
        .or(failure.element_id)
        .context("the first active semantic failure has no target element")?;
    for (element, edit) in edits {
        if *element != required {
            bail!(
                "the active deterministic repair plan permits semantic edits only for its first failing element {required}"
            );
        }
        if edit.source.is_some()
            && !failure
                .allowed_repair_fields
                .contains(&RepairField::SourceText)
        {
            bail!("the active deterministic failure does not permit source_text");
        }
        if edit.translation.is_some()
            && !failure
                .allowed_repair_fields
                .contains(&RepairField::TranslationText)
        {
            bail!("the active deterministic failure does not permit translation_text");
        }
    }
    Ok(())
}

fn validate_page_translation_targets(
    snapshot: &Snapshot,
    page: EntityId,
    elements: &[EntityId],
) -> Result<()> {
    snapshot.page(page)?;
    let mut logical_dialogues = Vec::new();
    if let Some(group) = snapshot.page(page)?.text_group()? {
        for layer in group.text_layers()? {
            if let Some(dialogue) = layer.content()?.logical_dialogue()? {
                logical_dialogues.push(dialogue);
            }
        }
    }
    for element in elements {
        if page_containing(snapshot, *element)? != page {
            bail!("page translation revision target {element} does not belong to page {page}");
        }
        if let Some(dialogue) = logical_dialogues.iter().find(|dialogue| {
            dialogue
                .members
                .iter()
                .any(|member| member.element_id == *element)
        }) && dialogue.members[0].element_id != *element
        {
            bail!(
                "page translation revision target {element} is a non-rendering logical dialogue member; target primary render element {}",
                dialogue.members[0].element_id
            );
        }
    }
    Ok(())
}

fn path_string(path: &Path) -> Result<String> {
    path.to_str()
        .context("path is not valid UTF-8")
        .map(ToOwned::to_owned)
}

fn paths_to_values(paths: &[PathBuf]) -> Result<Vec<String>> {
    paths.iter().map(|path| path_string(path)).collect()
}

fn effective_visibility(snapshot: &Snapshot, entity: EntityId) -> Result<(bool, f32)> {
    let mut visible = true;
    let mut opacity = 1.0;
    let mut current = Some(entity);
    while let Some(id) = current {
        if snapshot.page(id).is_ok() {
            break;
        }
        if let Some(value) = snapshot.component::<Visibility>(id)? {
            visible &= value.visible;
            opacity *= value.opacity;
        }
        current = snapshot.parent(id)?;
    }
    Ok((visible, opacity))
}

fn detected_panel_candidates(
    snapshot: &Snapshot,
    page: EntityId,
    raster_evidence: &BTreeMap<EntityId, RasterPanelEvidence>,
) -> Result<Vec<DetectedPanelCandidate>> {
    let mut candidates = Vec::new();
    for entity in snapshot.descendants(page)? {
        let Some(region) = entity.component::<koharu_scene::Region>()? else {
            continue;
        };
        if region.kind != PanelRegion::kind() {
            continue;
        }
        let Some(detection) = entity.component::<DetectionAnalysis>()? else {
            continue;
        };
        let Some(label) = detection
            .labels
            .iter()
            .filter(|label| label.kind == PanelRegion::kind())
            .max_by(|first, second| first.confidence.total_cmp(&second.confidence))
        else {
            continue;
        };
        let koharu_scene::Origin::Generated(generation) = detection.origin else {
            continue;
        };
        let geometry = entity
            .component::<Geometry>()?
            .map(element_geometry)
            .with_context(|| format!("detected panel {} has no geometry", entity.id()))?;
        let Some(raster_evidence) = raster_evidence.get(&entity.id()).cloned() else {
            continue;
        };
        let Some(source_geometry) =
            snapshot.component::<Geometry>(raster_evidence.source_relationship.source_region_id)?
        else {
            continue;
        };
        if generation.producer.as_str() != raster_evidence.detector.producer
            || generation.model.as_deref() != Some(raster_evidence.detector.version)
            || geometry.bounds != raster_evidence.panel_bbox
            || validate_raster_panel_evidence(
                &raster_evidence,
                raster_evidence.source_relationship.source_region_id,
                element_geometry(source_geometry).bounds,
            )
            .is_err()
        {
            continue;
        }
        candidates.push(DetectedPanelCandidate {
            region_id: entity.id(),
            region_kind: region.kind.as_str().to_owned(),
            geometry,
            detection_label: region.label.unwrap_or_else(|| "panel".to_owned()),
            detection_confidence: label.confidence,
            detector_producer: generation.producer.to_string(),
            detector_model: generation.model,
            raster_evidence,
        });
    }
    candidates.sort_by(|first, second| first.region_id.cmp(&second.region_id));
    Ok(candidates)
}

fn element_geometry(geometry: Geometry) -> ElementGeometry {
    let points = geometry
        .points
        .into_iter()
        .map(|point| ElementPoint {
            x: point.x,
            y: point.y,
        })
        .collect::<Vec<_>>();
    let bounds = points_bounds(&points);
    ElementGeometry { points, bounds }
}

fn render_bounds(bounds: koharu_renderer::RenderBounds) -> Option<ElementBounds> {
    let bounds = ElementBounds {
        x: f64::from(bounds.x),
        y: f64::from(bounds.y),
        width: f64::from(bounds.width),
        height: f64::from(bounds.height),
    };
    [bounds.x, bounds.y, bounds.width, bounds.height]
        .into_iter()
        .all(f64::is_finite)
        .then_some(bounds)
}

fn text_diagnostics(frame: &koharu_renderer::Frame, entity: EntityId) -> Vec<String> {
    frame
        .diagnostics()
        .iter()
        .filter_map(|diagnostic| match diagnostic {
            RenderDiagnostic::TextOverflow {
                entity: diagnostic_entity,
                ..
            } if *diagnostic_entity == entity => Some("text_overflow".to_owned()),
            RenderDiagnostic::TextBelowReadableSize {
                entity: diagnostic_entity,
                ..
            } if *diagnostic_entity == entity => Some("text_below_readable_size".to_owned()),
            _ => None,
        })
        .collect()
}

fn points_bounds(points: &[ElementPoint]) -> ElementBounds {
    let min_x = points
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min);
    let min_y = points
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min);
    let max_x = points
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_y = points
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max);
    if min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite() {
        ElementBounds {
            x: min_x,
            y: min_y,
            width: (max_x - min_x).max(0.0),
            height: (max_y - min_y).max(0.0),
        }
    } else {
        ElementBounds::default()
    }
}

fn bounds_intersect_page(bounds: ElementBounds, page_width: f64, page_height: f64) -> bool {
    bounds.width > 0.0
        && bounds.height > 0.0
        && bounds.x < page_width
        && bounds.y < page_height
        && bounds.x + bounds.width > 0.0
        && bounds.y + bounds.height > 0.0
}

fn repair_element_state(
    snapshot: &Snapshot,
    element: EntityId,
) -> Result<(EntityId, RepairElementState)> {
    let layer = snapshot.text_layer(element)?;
    let content = layer.content()?;
    let source = content.source()?.map(|value| RepairText {
        text: value.text.value,
        language: value.language.map(|language| language.to_string()),
        origin: value.text.origin,
    });
    let translation = content.translation()?.map(|value| RepairText {
        text: value.text.value,
        language: value.language.map(|language| language.to_string()),
        origin: value.text.origin,
    });
    Ok((
        content.id(),
        RepairElementState {
            source,
            translation,
            typography: layer.typography()?,
            layout_kind: layer.layout()?.kind,
            authored_layout_geometry: snapshot.component(element)?,
        },
    ))
}

fn changed_repair_fields(
    before: &RepairElementState,
    after: &RepairElementState,
) -> Vec<&'static str> {
    let mut fields = Vec::new();
    if !repair_text_equal(before.source.as_ref(), after.source.as_ref()) {
        fields.push("source_text");
    }
    if !repair_text_equal(before.translation.as_ref(), after.translation.as_ref()) {
        fields.push("translation_text");
    }
    if !typography_intent_equal(before.typography.as_ref(), after.typography.as_ref()) {
        fields.push("typography");
    }
    if before.layout_kind != after.layout_kind {
        fields.push("layout_kind");
    }
    if !geometry_intent_equal(
        before.authored_layout_geometry.as_ref(),
        after.authored_layout_geometry.as_ref(),
    ) {
        fields.push("layout_geometry");
    }
    fields
}

fn planned_repair_failure(
    plan: &DeterministicRepairPlan,
    requested_element: EntityId,
) -> Result<Option<&DeterministicRepairFailure>> {
    let Some(failure) = plan.blocking_failures.first() else {
        return Ok(None);
    };
    let Some(required_element) = failure.primary_render_element_id.or(failure.element_id) else {
        bail!(
            "the first unresolved deterministic failure {} has no revisable element; repair loop cannot continue",
            rejection_code_name(failure.code)
        );
    };
    if requested_element != required_element {
        bail!(
            "repair plan requires the first unresolved failing element {required_element}, but revise_element targeted {requested_element}"
        );
    }
    if failure.allowed_repair_fields.is_empty() && failure.next_action.is_none() {
        bail!(
            "the first unresolved deterministic failure {} on element {required_element} has no supported revise_element fields; repair loop cannot continue",
            rejection_code_name(failure.code)
        );
    }
    Ok(Some(failure))
}

fn validate_planned_repair_fields(
    failure: &DeterministicRepairFailure,
    changed_fields: &[&str],
) -> Result<()> {
    let changed = changed_fields
        .iter()
        .map(|field| match *field {
            "source_text" => RepairField::SourceText,
            "translation_text" => RepairField::TranslationText,
            "typography" => RepairField::Typography,
            "layout_kind" | "layout_geometry" => RepairField::Layout,
            field => unreachable!("unknown repair field {field}"),
        })
        .collect::<Vec<_>>();
    if changed
        .iter()
        .all(|field| failure.allowed_repair_fields.contains(field))
    {
        return Ok(());
    }
    bail!(
        "repair plan failure {} only allows fields {}, but the revision changed {}",
        rejection_code_name(failure.code),
        serde_json::to_string(&failure.allowed_repair_fields)?,
        serde_json::to_string(&changed)?,
    )
}

fn rejection_code_name(code: AcceptanceRejectionCode) -> String {
    serde_json::to_value(code)
        .expect("acceptance rejection code must serialize")
        .as_str()
        .expect("acceptance rejection code must serialize as a string")
        .to_owned()
}

fn valid_bounds(bounds: ElementBounds) -> bool {
    [bounds.x, bounds.y, bounds.width, bounds.height]
        .into_iter()
        .all(f64::is_finite)
        && bounds.width > 0.0
        && bounds.height > 0.0
}

fn repair_text_equal(first: Option<&RepairText>, second: Option<&RepairText>) -> bool {
    match (first, second) {
        (Some(first), Some(second)) => {
            first.text == second.text && first.language == second.language
        }
        (None, None) => true,
        _ => false,
    }
}

fn typography_intent_equal(first: Option<&Typography>, second: Option<&Typography>) -> bool {
    match (first, second) {
        (Some(first), Some(second)) => {
            let mut first = first.clone();
            let mut second = second.clone();
            first.origin = koharu_scene::Origin::User;
            second.origin = koharu_scene::Origin::User;
            first == second
        }
        (None, None) => true,
        _ => false,
    }
}

fn geometry_intent_equal(first: Option<&Geometry>, second: Option<&Geometry>) -> bool {
    match (first, second) {
        (Some(first), Some(second)) => first.points == second.points,
        (None, None) => true,
        _ => false,
    }
}

fn validate_revision_request(arguments: &ReviseElement) -> Result<()> {
    if arguments.reason.trim().is_empty() {
        bail!("revision reason cannot be empty");
    }
    if arguments.source_text.is_none()
        && arguments.translation_text.is_none()
        && arguments.typography.is_none()
        && arguments.layout.is_none()
    {
        bail!("revision must change source text, translation, typography, or layout");
    }
    for (label, text) in [
        ("source text", arguments.source_text.as_deref()),
        ("translation text", arguments.translation_text.as_deref()),
    ] {
        if text.is_some_and(|text| text.trim().is_empty()) {
            bail!("corrected {label} cannot be empty");
        }
    }
    if let Some(bounds) = arguments.layout.as_ref().and_then(|layout| layout.bounds) {
        if ![bounds.x, bounds.y, bounds.width, bounds.height]
            .into_iter()
            .all(f64::is_finite)
            || bounds.width <= 0.0
            || bounds.height <= 0.0
        {
            bail!("layout bounds must be finite with positive width and height");
        }
    }
    if let Some(layout) = &arguments.layout {
        if layout.bounds.is_some() && layout.padding.is_some() {
            bail!("layout bounds and padding are mutually exclusive");
        }
        if let Some(padding) = layout.padding {
            for (name, value) in [
                ("top", padding.top),
                ("right", padding.right),
                ("bottom", padding.bottom),
                ("left", padding.left),
            ] {
                if !value.is_finite() || value < 0.0 {
                    bail!("layout padding {name} must be finite and nonnegative");
                }
            }
        }
    }
    if let Some(typography) = &arguments.typography {
        if typography
            .size
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            bail!("typography size must be finite and positive");
        }
        if typography
            .stroke_width
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            bail!("typography stroke width must be finite and nonnegative");
        }
    }
    Ok(())
}

fn repair_layout_geometry(
    snapshot: &Snapshot,
    element: EntityId,
    content: EntityId,
    layout: &RepairLayout,
) -> Result<Option<Geometry>> {
    if let Some(bounds) = layout.bounds {
        return Ok(Some(Geometry::rectangle(
            bounds.x,
            bounds.y,
            bounds.width,
            bounds.height,
        )));
    }
    let Some(padding) = layout.padding else {
        return Ok(None);
    };
    let layer = snapshot.text_layer(element)?;
    let safe_region = match layer.balloon_target()? {
        Some(region) => Some(region),
        None => snapshot.text_content(content)?.source_region()?,
    }
    .context("layout padding requires a detected balloon or text-safe source region")?;
    let bounds = element_geometry(safe_region.geometry()?).bounds;
    let width = bounds.width - padding.left - padding.right;
    let height = bounds.height - padding.top - padding.bottom;
    if ![bounds.x, bounds.y, width, height]
        .into_iter()
        .all(f64::is_finite)
        || width <= 0.0
        || height <= 0.0
    {
        bail!("layout padding leaves no positive text frame inside the detected safe region");
    }
    Ok(Some(Geometry::rectangle(
        bounds.x + padding.left,
        bounds.y + padding.top,
        width,
        height,
    )))
}

fn validate_text_layout_request(arguments: &PreviewTextLayout) -> Result<()> {
    if arguments.reason.trim().is_empty() {
        bail!("text layout preview reason cannot be empty");
    }
    if let TextLayoutOptions::Controlled(options) = &arguments.options {
        if options.max_lines == 0 || options.max_lines > 12 {
            bail!("text layout max_lines must be between 1 and 12");
        }
        if options
            .font_scale
            .is_some_and(|scale| !scale.is_finite() || !(0.0..1.0).contains(&scale))
        {
            bail!(
                "font_scale must be finite, greater than zero, and less than one (decrease only)"
            );
        }
        if let Some(delta) = options.safe_padding_increase_px {
            validate_text_safe_inset_delta(delta)?;
        }
    }
    Ok(())
}

fn preview_text_layout_arguments(call: &ToolCall) -> Result<PreviewTextLayout> {
    let arguments = arguments(call)?;
    validate_text_layout_request(&arguments)?;
    Ok(arguments)
}

fn source_bound_interjection_fallback_target<'a>(
    snapshot: &Snapshot,
    element: EntityId,
    content: EntityId,
    strategy: SourceBoundInterjectionFallback,
    assessments: impl Iterator<Item = &'a FreeDialogueAnchorAssessment>,
    thresholds: &QualityThresholds,
) -> Result<Geometry> {
    if strategy != SourceBoundInterjectionFallback::NativeVertical {
        bail!("unsupported source-bound interjection fallback strategy");
    }
    let layer = snapshot.text_layer(element)?;
    if layer.balloon_target()?.is_some() || snapshot.relation_from::<FlowsIn>(element)?.is_some() {
        bail!("source-bound interjection fallback requires no container or target relation");
    }
    let content_ref = snapshot.text_content(content)?;
    let source_region = content_ref
        .source_region()?
        .context("source-bound interjection fallback requires its original source region")?;
    let source_bounds = element_geometry(source_region.geometry()?).bounds;
    let source = content_ref
        .source()?
        .context("source-bound interjection fallback requires source semantics")?;
    let target = content_ref
        .translation()?
        .context("source-bound interjection fallback requires target semantics")?;
    let role = content_ref
        .role()?
        .context("source-bound interjection fallback requires an exact source role")?;
    let mut matching_assessments =
        assessments.filter(|assessment| assessment.source_region_id == source_region.id());
    let assessment = matching_assessments.next().context(
        "source-bound interjection fallback requires exactly one source-raster assessment",
    )?;
    if matching_assessments.next().is_some() {
        bail!("source-bound interjection fallback requires exactly one source-raster assessment");
    }
    validate_source_bound_interjection_fallback(
        assessment,
        source_region.id(),
        source_bounds,
        &source.text.value,
        &target.text.value,
    )
    .map_err(|reason| {
        anyhow!("source-bound interjection fallback evidence is invalid: {reason}")
    })?;
    let typography = layer
        .typography()?
        .context("source-bound interjection fallback requires existing typography intent")?;
    let initial_horizontal_layout = typography.writing_mode == Some(WritingMode::Horizontal);
    let controlled_vertical_layout = typography.writing_mode == Some(WritingMode::Vertical)
        && typography.alignment == Some(TextAlignment::Center)
        && typography.size == Some(thresholds.min_rendered_font_size_px as f32)
        && !typography.auto_fit;
    if role.role != assessment.role_gate.source_scene_role
        || source.language.as_ref().map(ToString::to_string).as_deref()
            != Some(assessment.writing_mode_gate.source_language.as_str())
        || target.language.as_ref().map(ToString::to_string).as_deref()
            != Some(assessment.writing_mode_gate.target_language.as_str())
        || snapshot
            .component::<OcrAnalysis>(source_region.id())?
            .is_none_or(|analysis| analysis.direction != TextDirection::Vertical)
        || (!initial_horizontal_layout && !controlled_vertical_layout)
    {
        bail!("source-bound interjection fallback role, language, or writing-mode binding changed");
    }
    let fit = snapshot.relation_from::<FitsTo>(element)?.context(
        "source-bound interjection fallback requires the retained source FitsTo relation",
    )?;
    if fit.value().target != source_region.id() {
        bail!("source-bound interjection fallback no longer targets its source identity region");
    }
    let padding = thresholds.min_text_safe_padding_px;
    let inset = ElementBounds {
        x: source_bounds.x + padding,
        y: source_bounds.y + padding,
        width: source_bounds.width - padding * 2.0,
        height: source_bounds.height - padding * 2.0,
    };
    let required_width = thresholds.min_rendered_font_size_px;
    let required_height =
        thresholds.min_rendered_font_size_px * visible_grapheme_units(&target.text.value) as f64;
    if ![inset.x, inset.y, inset.width, inset.height]
        .into_iter()
        .all(f64::is_finite)
        || inset.width + f64::EPSILON < required_width
        || inset.height + f64::EPSILON < required_height
    {
        bail!(
            "source-bound interjection fallback cannot provide {:.2}px font and {:.2}px clearance inside source bounds",
            thresholds.min_rendered_font_size_px,
            padding
        );
    }
    let mut geometry = source_region.geometry()?;
    geometry.origin = koharu_scene::Origin::User;
    Ok(geometry)
}

fn source_bound_interjection_render_fit(
    source_geometry: &ElementGeometry,
    glyph_ink: &GlyphInkMask,
    required_clearance_px: f64,
) -> Result<Geometry> {
    if source_geometry.points.len() < 3
        || !valid_bounds(source_geometry.bounds)
        || !required_clearance_px.is_finite()
        || required_clearance_px <= 0.0
        || glyph_ink.width == 0
        || glyph_ink.height == 0
        || !usize::try_from(u64::from(glyph_ink.width) * u64::from(glyph_ink.height))
            .is_ok_and(|pixels| pixels == glyph_ink.alpha.len())
    {
        bail!("source-bound interjection render fit requires finite source, layout, and glyph ink");
    }
    let signed_area = source_geometry
        .points
        .iter()
        .zip(source_geometry.points.iter().cycle().skip(1))
        .take(source_geometry.points.len())
        .map(|(first, second)| first.x * second.y - second.x * first.y)
        .sum::<f64>();
    if !signed_area.is_finite() || signed_area.abs() <= f64::EPSILON {
        bail!("source-bound interjection source polygon has no finite interior");
    }
    let orientation = signed_area.signum();
    for index in 0..source_geometry.points.len() {
        let first = source_geometry.points[index];
        let second = source_geometry.points[(index + 1) % source_geometry.points.len()];
        let third = source_geometry.points[(index + 2) % source_geometry.points.len()];
        let turn = (second.x - first.x) * (third.y - second.y)
            - (second.y - first.y) * (third.x - second.x);
        if !turn.is_finite() || turn * orientation < -f64::EPSILON {
            bail!("source-bound interjection source polygon must be convex");
        }
    }

    let mut constraints = Vec::with_capacity(source_geometry.points.len());
    for (first, second) in source_geometry
        .points
        .iter()
        .zip(source_geometry.points.iter().cycle().skip(1))
        .take(source_geometry.points.len())
    {
        let edge_x = second.x - first.x;
        let edge_y = second.y - first.y;
        let edge_length = edge_x.hypot(edge_y);
        if !edge_length.is_finite() || edge_length <= f64::EPSILON {
            bail!("source-bound interjection source polygon contains a degenerate edge");
        }
        let normal_x = orientation * -edge_y / edge_length;
        let normal_y = orientation * edge_x / edge_length;
        let mut minimum_projection = f64::INFINITY;
        for (index, alpha) in glyph_ink.alpha.iter().enumerate() {
            if *alpha == 0 {
                continue;
            }
            let x = index % glyph_ink.width as usize;
            let y = index / glyph_ink.width as usize;
            let point_x = f64::from(glyph_ink.left) + x as f64 + 0.5;
            let point_y = f64::from(glyph_ink.top) + y as f64 + 0.5;
            minimum_projection = minimum_projection
                .min(normal_x * (point_x - first.x) + normal_y * (point_y - first.y));
        }
        if !minimum_projection.is_finite() {
            bail!("source-bound interjection final render contains no glyph ink");
        }
        constraints.push((
            normal_x,
            normal_y,
            required_clearance_px + PIXEL_HALF_DIAGONAL - minimum_projection,
        ));
    }

    let satisfies = |x: f64, y: f64| {
        constraints.iter().all(|(normal_x, normal_y, minimum)| {
            normal_x.mul_add(x, normal_y * y) + f64::EPSILON >= *minimum
        })
    };
    let mut candidates = vec![(0.0, 0.0)];
    for &(normal_x, normal_y, minimum) in &constraints {
        let squared_length = normal_x.mul_add(normal_x, normal_y * normal_y);
        if squared_length > f64::EPSILON {
            candidates.push((
                normal_x * minimum / squared_length,
                normal_y * minimum / squared_length,
            ));
        }
    }
    for first in 0..constraints.len() {
        for second in first + 1..constraints.len() {
            let (first_x, first_y, first_minimum) = constraints[first];
            let (second_x, second_y, second_minimum) = constraints[second];
            let determinant = first_x * second_y - first_y * second_x;
            if determinant.abs() <= f64::EPSILON {
                continue;
            }
            candidates.push((
                (first_minimum * second_y - first_y * second_minimum) / determinant,
                (first_x * second_minimum - first_minimum * second_x) / determinant,
            ));
        }
    }
    let (offset_x, offset_y) = candidates
        .into_iter()
        .filter(|(x, y)| x.is_finite() && y.is_finite() && satisfies(*x, *y))
        .min_by(|(first_x, first_y), (second_x, second_y)| {
            first_x
                .mul_add(*first_x, first_y * first_y)
                .total_cmp(&second_x.mul_add(*second_x, second_y * second_y))
        })
        .context(
            "source-bound interjection glyph ink cannot satisfy the immutable source contour",
        )?;
    let [top_left, top_right, bottom_right, bottom_left] = source_geometry.points.as_slice() else {
        bail!("source-bound interjection render fit requires a four-point source frame");
    };
    let top = (top_right.x - top_left.x, top_right.y - top_left.y);
    let right = (bottom_right.x - top_right.x, bottom_right.y - top_right.y);
    let bottom_edge = (
        bottom_left.x - bottom_right.x,
        bottom_left.y - bottom_right.y,
    );
    let left_edge = (top_left.x - bottom_left.x, top_left.y - bottom_left.y);
    let width = top.0.hypot(top.1);
    let height = right.0.hypot(right.1);
    let tolerance = width.max(height) * 1e-6;
    if width <= f64::EPSILON
        || height <= f64::EPSILON
        || (top.0 * right.0 + top.1 * right.1).abs() > width * height * 1e-6
        || (bottom_edge.0 + top.0).hypot(bottom_edge.1 + top.1) > tolerance
        || (left_edge.0 + right.0).hypot(left_edge.1 + right.1) > tolerance
    {
        bail!("source-bound interjection render fit requires a rectangular source frame");
    }
    let horizontal = offset_x * top.0 / width + offset_y * top.1 / width;
    let vertical = offset_x * right.0 / height + offset_y * right.1 / height;
    let left = horizontal.max(0.0) * 2.0;
    let right_inset = (-horizontal).max(0.0) * 2.0;
    let top_inset = vertical.max(0.0) * 2.0;
    let bottom = (-vertical).max(0.0) * 2.0;
    if left + right_inset >= width || top_inset + bottom >= height {
        bail!("source-bound interjection measured inset leaves no positive layout frame");
    }
    let horizontal_unit = (top.0 / width, top.1 / width);
    let vertical_unit = (right.0 / height, right.1 / height);
    let fitted = [
        (
            top_left.x + horizontal_unit.0 * left + vertical_unit.0 * top_inset,
            top_left.y + horizontal_unit.1 * left + vertical_unit.1 * top_inset,
        ),
        (
            top_right.x - horizontal_unit.0 * right_inset + vertical_unit.0 * top_inset,
            top_right.y - horizontal_unit.1 * right_inset + vertical_unit.1 * top_inset,
        ),
        (
            bottom_right.x - horizontal_unit.0 * right_inset - vertical_unit.0 * bottom,
            bottom_right.y - horizontal_unit.1 * right_inset - vertical_unit.1 * bottom,
        ),
        (
            bottom_left.x + horizontal_unit.0 * left - vertical_unit.0 * bottom,
            bottom_left.y + horizontal_unit.1 * left - vertical_unit.1 * bottom,
        ),
    ];
    let mut geometry = Geometry::rectangle(0.0, 0.0, 1.0, 1.0);
    for (point, (x, y)) in geometry.points.iter_mut().zip(fitted) {
        point.x = x;
        point.y = y;
    }
    Ok(geometry)
}

fn source_bound_interjection_typography(
    existing: Option<&Typography>,
    minimum_font_size_px: f64,
) -> Result<Typography> {
    let mut typography = existing
        .cloned()
        .context("source-bound interjection fallback requires existing typography intent")?;
    if !minimum_font_size_px.is_finite()
        || minimum_font_size_px <= 0.0
        || minimum_font_size_px > f64::from(f32::MAX)
    {
        bail!("source-bound interjection fallback minimum font is invalid");
    }
    typography.origin = koharu_scene::Origin::User;
    typography.size = Some(minimum_font_size_px as f32);
    typography.auto_fit = false;
    typography.alignment = Some(TextAlignment::Center);
    typography.writing_mode = Some(WritingMode::Vertical);
    Ok(typography)
}

#[derive(Clone, Copy)]
enum ControlledTargetRelation {
    FitsTo,
    FlowsIn,
}

#[derive(Clone)]
struct ControlledTargetMigration {
    target: EntityId,
    relation_to_remove: koharu_scene::RelationId,
    source_region: EntityId,
    add_inside: bool,
    relation: ControlledTargetRelation,
    layout_geometry: Option<Geometry>,
    layout_typography: Option<Typography>,
}

fn apply_controlled_target_migration(
    edit: &mut koharu_scene::Edit,
    element: EntityId,
    migration: ControlledTargetMigration,
) -> koharu_scene::Result<()> {
    edit.remove_relation(migration.relation_to_remove)?;
    match migration.relation {
        ControlledTargetRelation::FitsTo => {
            edit.relate::<FitsTo>(element, migration.target)?;
        }
        ControlledTargetRelation::FlowsIn => {
            edit.relate::<FlowsIn>(element, migration.target)?;
        }
    }
    if let Some(geometry) = &migration.layout_geometry {
        // This renderer frame is derived from the detector-owned target. RecognizedFrom remains
        // on semantic content so source identity and pixels do not compete with target placement.
        edit.set(element, geometry)?;
    }
    if let Some(typography) = &migration.layout_typography {
        edit.set(element, typography)?;
    }
    if migration.add_inside {
        edit.relate::<Inside>(migration.source_region, migration.target)?;
    }
    Ok(())
}

fn validate_source_bound_interjection_changes(
    before: &RepairElementState,
    after: &RepairElementState,
) -> Result<()> {
    if before.source != after.source
        || before.translation != after.translation
        || before.layout_kind != after.layout_kind
    {
        bail!("source-bound interjection fallback cannot change semantics or layout ownership");
    }
    let typography = after
        .typography
        .as_ref()
        .context("source-bound interjection fallback typography is missing")?;
    if typography.writing_mode != Some(WritingMode::Vertical)
        || typography.alignment != Some(TextAlignment::Center)
        || typography.auto_fit
    {
        bail!("source-bound interjection fallback typography is not the controlled vertical form");
    }
    Ok(())
}

fn verified_ui_panel_layout_target(
    snapshot: &Snapshot,
    element: EntityId,
    content: EntityId,
    decision: Option<&UiPanelAnchorDecision>,
) -> Result<Option<EntityId>> {
    let Some(decision) = decision else {
        return Ok(None);
    };
    if decision.element_id != element
        || decision.content_id != content
        || snapshot
            .component::<TextRole>(content)?
            .is_none_or(|role| role.role != UI_TEXT_ROLE)
    {
        bail!("verified UI-panel anchor identity or UI role no longer matches the scene");
    }
    let source = snapshot
        .text_content(content)?
        .source_region()?
        .context("verified UI-panel anchor source region is missing")?;
    if source.id() != decision.source_region_id {
        bail!("verified UI-panel anchor source ownership changed");
    }
    let panel = snapshot.analysis_region(decision.panel.region_id)?;
    let region = panel.region()?;
    let detection = panel
        .detection()?
        .context("verified UI-panel target no longer has detection provenance")?;
    if region.kind != PanelRegion::kind() {
        bail!("verified UI-panel target no longer has panel region kind");
    }
    if !detection
        .labels
        .iter()
        .any(|label| label.kind == PanelRegion::kind())
    {
        bail!("verified UI-panel target no longer has a panel detection label");
    }
    let koharu_scene::Origin::Generated(generation) = &detection.origin else {
        bail!("verified UI-panel target no longer has generated detector provenance");
    };
    let panel_geometry = element_geometry(panel.geometry()?);
    if !valid_element_geometry(&panel_geometry) {
        bail!("verified UI-panel target no longer has finite positive panel geometry");
    }
    if panel_geometry != decision.panel.geometry {
        bail!("verified UI-panel target geometry changed after evidence binding");
    }
    let raster_evidence = &decision.panel.raster_evidence;
    if generation.producer.as_str() != raster_evidence.detector.producer
        || generation.model.as_deref() != Some(raster_evidence.detector.version)
        || decision.panel.detector_producer != raster_evidence.detector.producer
        || decision.panel.detector_model.as_deref() != Some(raster_evidence.detector.version)
    {
        bail!("verified UI-panel detector provenance no longer matches its raster evidence");
    }
    raster_candidate_supports_ui_role(
        raster_evidence,
        source.id(),
        element_geometry(source.geometry()?).bounds,
        &decision.evidence.ui_role_and_function,
    )
    .map_err(|reason| anyhow!("verified UI-panel raster evidence is invalid: {reason}"))?;
    if decision.confidence < MINIMUM_UI_PANEL_VERIFICATION_CONFIDENCE
        || decision.source_containment_ratio < 0.90
        || decision.association_reason.trim().is_empty()
    {
        bail!("verified UI-panel association evidence is weak or incomplete");
    }
    if let Some(flow) = snapshot.relation_from::<FlowsIn>(element)? {
        if flow.value().target != panel.id() {
            bail!("text element already flows into a different target");
        }
        return Ok(None);
    }
    let fit = snapshot
        .relation_from::<FitsTo>(element)?
        .context("verified UI-panel text must remain source-bound before controlled migration")?;
    if fit.value().target != source.id() {
        bail!("verified UI-panel text no longer fits its retained source identity region");
    }
    Ok(Some(panel.id()))
}

fn verified_free_dialogue_layout_target(
    snapshot: &Snapshot,
    element: EntityId,
    content: EntityId,
    decision: Option<&FreeDialogueAnchorDecision>,
) -> Result<Option<EntityId>> {
    let Some(decision) = decision else {
        return Ok(None);
    };
    if decision.element_id != element || decision.content_id != content {
        bail!("free-dialogue anchor identity no longer matches the scene");
    }
    let content_ref = snapshot.text_content(content)?;
    let source_region = content_ref
        .source_region()?
        .context("free-dialogue anchor source region is missing")?;
    if source_region.id() != decision.source_region_id {
        bail!("free-dialogue anchor source ownership changed");
    }
    let source = content_ref
        .source()?
        .context("free-dialogue anchor source semantics are missing")?;
    let translation = content_ref
        .translation()?
        .context("free-dialogue anchor target translation is missing")?;
    let role = content_ref
        .role()?
        .context("free-dialogue anchor exact source role is missing")?;
    if role.role != decision.role_gate.source_scene_role
        || !matches!(
            role.role.as_str(),
            crate::free_dialogue::DIALOGUE_ROLE | crate::free_dialogue::FREE_DIALOGUE_SOURCE_ROLE
        )
    {
        bail!("free-dialogue anchor exact role gate no longer matches the scene");
    }
    validate_free_dialogue_anchor_decision(
        decision,
        element,
        content,
        source_region.id(),
        element_geometry(source_region.geometry()?).bounds,
        &source.text.value,
        &translation.text.value,
    )
    .map_err(|reason| anyhow!("free-dialogue anchor evidence is invalid: {reason}"))?;
    let target = snapshot.analysis_region(decision.target_region_id)?;
    if target.region()?.kind != TextRegion::kind()
        || element_geometry(target.geometry()?).bounds != decision.candidate_bounds
    {
        bail!("free-dialogue target region geometry or kind changed after raster binding");
    }
    let detection = target
        .detection()?
        .context("free-dialogue target no longer has detector provenance")?;
    let koharu_scene::Origin::Generated(generation) = detection.origin else {
        bail!("free-dialogue target no longer has generated detector provenance");
    };
    if generation.producer.as_str() != RASTER_FREE_DIALOGUE_DETECTOR
        || generation.model.as_deref() != Some(RASTER_FREE_DIALOGUE_DETECTOR_VERSION)
        || !detection
            .labels
            .iter()
            .any(|label| label.kind.as_str() == ADJACENT_FREE_DIALOGUE_DETECTION_KIND)
    {
        bail!("free-dialogue target detector provenance changed");
    }
    if snapshot.relation_from::<FlowsIn>(element)?.is_some() {
        bail!("free-dialogue text cannot use a bubble-only flows-in relation");
    }
    let fit = snapshot.relation_from::<FitsTo>(element)?.context(
        "free-dialogue text must remain source-bound before controlled preview migration",
    )?;
    if fit.value().target == target.id() {
        return Ok(None);
    }
    if fit.value().target != source_region.id() {
        bail!("free-dialogue text no longer fits its retained source identity region");
    }
    Ok(Some(target.id()))
}

fn validate_compact_translation_request(
    arguments: &PreviewCompactTranslation,
    target_language: Language,
) -> Result<()> {
    if arguments.rationale.trim().is_empty() {
        bail!("compact translation rationale cannot be empty");
    }
    if target_language != Language::Korean || target_language.tag() != "ko-KR" {
        bail!("preview_compact_translation is available only for a ko-KR target project");
    }
    if arguments.candidate_translation.text.trim().is_empty()
        || visible_grapheme_units(&arguments.candidate_translation.text) == 0
    {
        bail!("compact translation candidate must contain nonempty visible Korean text");
    }
    if arguments.candidate_translation.language != "ko-KR" {
        bail!(
            "compact translation candidate language must be ko-KR, got {}",
            arguments.candidate_translation.language
        );
    }
    Ok(())
}

fn visible_grapheme_units(text: &str) -> usize {
    text.graphemes(true)
        .filter(|grapheme| {
            grapheme
                .chars()
                .any(|character| !character.is_whitespace() && !character.is_control())
        })
        .count()
}

fn validate_compact_translation_length(current: &str, candidate: &str) -> Result<(usize, usize)> {
    let current_units = visible_grapheme_units(current);
    let candidate_units = visible_grapheme_units(candidate);
    if candidate_units == 0 {
        bail!("compact translation candidate must contain visible grapheme units");
    }
    if candidate_units >= current_units {
        bail!(
            "compact translation must contain strictly fewer visible grapheme units than the current translation ({candidate_units} >= {current_units}); no mutation made"
        );
    }
    Ok((current_units, candidate_units))
}

fn validate_compact_translation_plan(
    failure: &DeterministicRepairFailure,
    requested_element: EntityId,
    requested_group: EntityId,
) -> Result<()> {
    let required_element = failure.primary_render_element_id.or(failure.element_id);
    if required_element != Some(requested_element) {
        bail!(
            "compact translation must target the active primary render element {}",
            required_element
                .map(|element| element.to_string())
                .unwrap_or_else(|| "<missing>".to_owned())
        );
    }
    if failure.logical_group_id != Some(requested_group) || failure.member_ordinal_ids.len() < 2 {
        bail!("compact translation must target the active logical dialogue group exactly");
    }
    let eligible_failure = failure.code == AcceptanceRejectionCode::RenderedFontSizeBelowMinimum
        || is_long_dialogue_target_anchor_failure(failure)
        || is_long_dialogue_clearance_failure(failure);
    if !eligible_failure {
        bail!(
            "compact translation is not permitted for the active {} layout failure",
            rejection_code_name(failure.code)
        );
    }
    if failure.next_action.as_ref().map(|action| action.tool) != Some("preview_compact_translation")
    {
        bail!(
            "compact translation is not exposed until controlled layout is infeasible for the active group"
        );
    }
    Ok(())
}

fn compact_translation_evidence(
    observations: &PageSemanticEvidenceState,
    current_revision: Revision,
    page: EntityId,
    reference: &PageTranslationEvidenceReference,
    members: &[crate::repair::RepairLogicalDialogueMember],
) -> Result<RevisionEvidence> {
    if Revision::new(reference.scene_revision) != current_revision {
        bail!(
            "preview_compact_translation evidence revision does not match current revision {current_revision}"
        );
    }
    let evidence = validate_four_part_page_evidence(
        observations,
        current_revision,
        page,
        &reference.source_evidence_dossier_blake3,
        &reference.source_debug_artifact_blake3,
        &reference.dossier_blake3,
        &reference.debug_artifact_blake3,
        "preview_compact_translation",
    )?;
    compact_member_evidence(observations, page, members, "preview_compact_translation")?;
    Ok(RevisionEvidence::PageTranslationVisualEvidence {
        revision: current_revision,
        page_id: page,
        source_evidence_dossier_blake3: evidence.source_evidence_dossier_blake3,
        source_debug_artifact_blake3: evidence.source_debug_artifact_blake3,
        dossier_blake3: evidence.dossier_blake3,
        debug_artifact_blake3: evidence.debug_artifact_blake3,
    })
}

fn compact_member_evidence(
    observations: &PageSemanticEvidenceState,
    page: EntityId,
    members: &[crate::repair::RepairLogicalDialogueMember],
    action: &str,
) -> Result<Vec<CompactedTranslationMemberEvidence>> {
    let dossier = observations
        .page(page)
        .with_context(|| format!("{action} has no evidence for page {page}"))?
        .source_dossier
        .as_ref()
        .with_context(|| format!("{action} requires fresh inspect_source_evidence evidence"))?;
    members
        .iter()
        .map(|member| {
            let evidence = dossier
                .element_crops
                .get(&member.element_id)
                .with_context(|| {
                    format!(
                        "{action} requires original crop evidence for logical group member {}",
                        member.element_id
                    )
                })?;
            Ok(CompactedTranslationMemberEvidence {
                ordinal: member.ordinal,
                element: member.element_id.to_string(),
                source_crop_blake3: evidence.crop_blake3.clone(),
            })
        })
        .collect()
}

fn validate_prepared_compact_evidence(
    observations: &PageSemanticEvidenceState,
    current_revision: Revision,
    evidence: &RevisionEvidence,
    members: &[crate::repair::RepairLogicalDialogueMember],
) -> Result<()> {
    let RevisionEvidence::PageTranslationVisualEvidence {
        revision,
        page_id,
        source_evidence_dossier_blake3,
        source_debug_artifact_blake3,
        dossier_blake3,
        debug_artifact_blake3,
    } = evidence
    else {
        bail!("compact translation preview does not retain four-part page evidence");
    };
    if *revision != current_revision {
        bail!("compact translation preview evidence is stale");
    }
    validate_four_part_page_evidence(
        observations,
        current_revision,
        *page_id,
        source_evidence_dossier_blake3,
        source_debug_artifact_blake3,
        dossier_blake3,
        debug_artifact_blake3,
        "commit_compact_translation",
    )?;
    compact_member_evidence(
        observations,
        *page_id,
        members,
        "commit_compact_translation",
    )?;
    Ok(())
}

fn validate_compact_translation_change(
    before: &RepairElementState,
    after: &RepairElementState,
) -> Result<()> {
    let changed = changed_repair_fields(before, after);
    if changed != ["translation_text"] {
        bail!(
            "preview_compact_translation may change only translation text; attempted {}",
            serde_json::to_string(&changed)?
        );
    }
    if before.source != after.source
        || before.typography != after.typography
        || before.layout_kind != after.layout_kind
        || !geometry_intent_equal(
            before.authored_layout_geometry.as_ref(),
            after.authored_layout_geometry.as_ref(),
        )
    {
        bail!(
            "compact translation attempted an unsupported source, typography, or layout mutation"
        );
    }
    Ok(())
}

fn validate_logical_group_unchanged(
    before: &ProjectInspection,
    after: &ProjectInspection,
    page: EntityId,
    logical_group: EntityId,
    primary: EntityId,
    expected_members: &[crate::repair::RepairLogicalDialogueMember],
) -> Result<()> {
    let before_group = before
        .project
        .pages
        .iter()
        .find(|candidate| candidate.id == page)
        .and_then(|candidate| {
            candidate
                .logical_dialogue_groups
                .iter()
                .find(|group| group.group_id == logical_group)
        })
        .context("active logical dialogue group is missing")?;
    let after_group = after
        .project
        .pages
        .iter()
        .find(|candidate| candidate.id == page)
        .and_then(|candidate| {
            candidate
                .logical_dialogue_groups
                .iter()
                .find(|group| group.group_id == logical_group)
        })
        .context("candidate logical dialogue group is missing")?;
    let actual_members = |group: &LogicalDialogueGroupInspection| {
        group
            .members
            .iter()
            .map(|member| crate::repair::RepairLogicalDialogueMember {
                ordinal: member.ordinal,
                element_id: member.element_id,
                source_region_id: member.source_region_id,
            })
            .collect::<Vec<_>>()
    };
    if before_group.primary_render_element_id != primary
        || after_group.primary_render_element_id != primary
        || actual_members(before_group) != expected_members
        || actual_members(after_group) != expected_members
        || before_group.logical_source_text != after_group.logical_source_text
    {
        bail!("compact translation cannot change source/OCR or logical group member ordering");
    }
    Ok(())
}

fn validate_text_layout_plan(
    failure: Option<&DeterministicRepairFailure>,
    options: &TextLayoutOptions,
) -> Result<()> {
    let Some(failure) = failure else {
        return Ok(());
    };
    let planned_text_layout_preview = failure
        .next_action
        .as_ref()
        .is_some_and(|action| action.tool == "preview_text_layout");
    if failure.code == AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior
        && !planned_text_layout_preview
    {
        bail!(
            "text-safe clearance uses preview_increase_text_safe_padding and commit_text_safe_layout_repair, not preview_text_layout"
        );
    }
    if !planned_text_layout_preview
        && !failure
            .allowed_repair_fields
            .iter()
            .any(|field| matches!(field, RepairField::Typography | RepairField::Layout))
    {
        bail!(
            "the first active deterministic failure is semantic; revise its permitted semantic field before layout"
        );
    }
    if failure
        .next_action
        .as_ref()
        .is_some_and(|action| action.tool != "preview_text_layout")
    {
        bail!(
            "the active deterministic failure requires {}; preview_text_layout is not permitted",
            failure.next_action.as_ref().unwrap().tool
        );
    }
    if let Some(operation) = failure
        .next_action
        .as_ref()
        .filter(|action| action.tool == "preview_text_layout")
        .map(|action| action.operation)
    {
        let fallback_requested = options.source_bound_interjection_fallback().is_some();
        let fallback_planned =
            operation == "controlled_source_bound_vertical_interjection_fallback";
        if fallback_requested != fallback_planned {
            bail!(
                "preview_text_layout options do not match the active deterministic operation {operation}"
            );
        }
    }
    Ok(())
}

fn host_deterministic_repair(
    failure: &DeterministicRepairFailure,
) -> Result<Option<HostDeterministicRepair>> {
    let Some(action) = failure.next_action.as_ref() else {
        return Ok(None);
    };
    if action.required_evidence_revision != failure.required_evidence_revision
        || failure.primary_render_element_id.or(failure.element_id) != Some(action.element)
    {
        bail!("host deterministic repair action does not match its reviewed failure");
    }
    match (action.tool, action.operation) {
        (
            "preview_text_layout",
            operation @ ("controlled_verified_ui_panel_target_layout"
            | "controlled_source_raster_verified_free_dialogue_target_layout"),
        ) => Ok(Some(HostDeterministicRepair::TextLayout {
            element: action.element,
            options: TextLayoutOptions::Controlled(ControlledTextLayoutOptions {
                line_break_policy: KoreanLineBreakPolicy::PreserveExisting,
                max_lines: 12,
                alignment: None,
                font_scale: None,
                safe_padding_increase_px: None,
            }),
            operation,
        })),
        (
            "preview_text_layout",
            operation @ "controlled_source_bound_vertical_interjection_fallback",
        ) => Ok(Some(HostDeterministicRepair::TextLayout {
            element: action.element,
            options: TextLayoutOptions::SourceBoundInterjectionFallback(
                SourceBoundInterjectionFallbackOptions {
                    source_bound_interjection_fallback:
                        SourceBoundInterjectionFallback::NativeVertical,
                },
            ),
            operation,
        })),
        ("preview_text_layout", "controlled_text_layout") => Ok(None),
        ("preview_increase_text_safe_padding", "increase_text_safe_padding") => {
            let required = failure
                .expected
                .get("minimum_clearance_px")
                .and_then(Value::as_f64)
                .context("host clearance repair is missing its reviewed required clearance")?;
            let current = failure
                .actual
                .get("minimum_clearance_px")
                .and_then(Value::as_f64)
                .context("host clearance repair is missing its reviewed current clearance")?;
            if !required.is_finite() || !current.is_finite() || required <= 0.0 {
                bail!("host clearance repair has non-finite or non-positive reviewed inputs");
            }
            let delta = (required - current).ceil().clamp(1.0, required);
            Ok(Some(HostDeterministicRepair::TextSafeClearance {
                element: action.element,
                inset_delta_px: TextSafeInsetDelta {
                    top: delta,
                    right: delta,
                    bottom: delta,
                    left: delta,
                },
            }))
        }
        ("preview_compact_translation", _) => Ok(None),
        _ => bail!(
            "repair action {} / {} is not an allowed host-recorded deterministic repair",
            action.tool,
            action.operation
        ),
    }
}

fn host_deterministic_repair_trace(repair: &HostDeterministicRepair) -> Value {
    match repair {
        HostDeterministicRepair::TextLayout {
            element,
            options,
            operation,
        } => json!({
            "kind": "text_layout",
            "operation": operation,
            "element": element,
            "options": options,
        }),
        HostDeterministicRepair::TextSafeClearance {
            element,
            inset_delta_px,
        } => json!({
            "kind": "text_safe_clearance",
            "operation": "increase_text_safe_padding",
            "element": element,
            "inset_delta_px": inset_delta_px,
        }),
    }
}

fn record_repair_action_attempt(
    attempted: &mut BTreeSet<RepairActionIdentity>,
    identity: RepairActionIdentity,
) -> bool {
    attempted.insert(identity)
}

fn apply_line_break_policy(
    text: &str,
    policy: KoreanLineBreakPolicy,
    max_lines: u8,
) -> Result<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        bail!("text layout requires a nonempty current translation");
    }
    if matches!(policy, KoreanLineBreakPolicy::PreserveExisting) {
        return Ok(trimmed.to_owned());
    }
    let words = trimmed.split_whitespace().collect::<Vec<_>>();
    if words.len() <= 1 || max_lines == 1 {
        return Ok(words.join(" "));
    }
    let line_count = usize::from(max_lines).min(words.len());
    let total_chars = words.iter().map(|word| word.chars().count()).sum::<usize>()
        + words.len().saturating_sub(1);
    let mut lines = Vec::with_capacity(line_count);
    let mut word_index = 0;
    for line_index in 0..line_count {
        let remaining_lines = line_count - line_index;
        let remaining_words = words.len() - word_index;
        let target = total_chars.div_ceil(line_count);
        let mut line = Vec::new();
        let mut chars = 0;
        while word_index < words.len() {
            let word = words[word_index];
            let addition = word.chars().count() + usize::from(!line.is_empty());
            if !line.is_empty()
                && chars + addition > target
                && remaining_words.saturating_sub(line.len()) >= remaining_lines
            {
                break;
            }
            line.push(word);
            chars += addition;
            word_index += 1;
            if words.len() - word_index == remaining_lines.saturating_sub(1) {
                break;
            }
        }
        lines.push(line.join(" "));
    }
    if word_index < words.len() {
        let suffix = words[word_index..].join(" ");
        if let Some(last) = lines.last_mut() {
            if !last.is_empty() {
                last.push(' ');
            }
            last.push_str(&suffix);
        }
    }
    Ok(lines.join("\n"))
}

fn candidate_text_layout_typography(
    existing: Option<&Typography>,
    options: &TextLayoutOptions,
) -> Result<Option<Typography>> {
    let TextLayoutOptions::Controlled(options) = options else {
        return Ok(None);
    };
    if options.alignment.is_none() && options.font_scale.is_none() {
        return Ok(None);
    }
    let mut typography = existing
        .cloned()
        .context("alignment and font scaling require existing typography intent")?;
    typography.origin = koharu_scene::Origin::User;
    if let Some(alignment) = options.alignment {
        typography.alignment = Some(alignment.into());
    }
    if let Some(scale) = options.font_scale {
        let size = typography
            .size
            .context("font_scale requires an explicit current typography size")?;
        typography.size = Some(size * scale);
        typography.auto_fit = false;
    }
    Ok(Some(typography))
}

fn controlled_text_layout_changes(
    before: &RepairElementState,
    after: &RepairElementState,
) -> Result<Vec<&'static str>> {
    let mut changes = Vec::new();
    if !repair_text_equal(before.translation.as_ref(), after.translation.as_ref()) {
        let before_text = before
            .translation
            .as_ref()
            .context("missing original translation")?;
        let after_text = after
            .translation
            .as_ref()
            .context("missing candidate translation")?;
        if normalize_layout_text(&before_text.text) != normalize_layout_text(&after_text.text)
            || before_text.language != after_text.language
        {
            bail!("preview_text_layout may only change translation whitespace and line breaks");
        }
        changes.push("translation_line_breaks");
    }
    if !typography_intent_equal(before.typography.as_ref(), after.typography.as_ref()) {
        let before_typography = before
            .typography
            .as_ref()
            .context("missing original typography")?;
        let after_typography = after
            .typography
            .as_ref()
            .context("missing candidate typography")?;
        let mut controlled_before = before_typography.clone();
        let mut controlled_after = after_typography.clone();
        controlled_before.origin = koharu_scene::Origin::User;
        controlled_after.origin = koharu_scene::Origin::User;
        let alignment_changed = controlled_before.alignment != controlled_after.alignment;
        let size_changed = controlled_before.size != controlled_after.size
            || controlled_before.auto_fit != controlled_after.auto_fit;
        controlled_before.alignment = controlled_after.alignment;
        controlled_before.size = controlled_after.size;
        controlled_before.auto_fit = controlled_after.auto_fit;
        if controlled_before != controlled_after {
            bail!("preview_text_layout attempted an unsupported typography mutation");
        }
        if alignment_changed {
            changes.push("typography_alignment");
        }
        if size_changed {
            changes.push("font_scale_decrease");
        }
    }
    if !geometry_intent_equal(
        before.authored_layout_geometry.as_ref(),
        after.authored_layout_geometry.as_ref(),
    ) {
        changes.push("safe_padding_increase");
    }
    if before.source != after.source || before.layout_kind != after.layout_kind {
        bail!("preview_text_layout attempted a semantic source or layout-kind mutation");
    }
    Ok(changes)
}

fn normalize_layout_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn text_layout_metrics_for(
    acceptance: &AcceptanceRecord,
    element: EntityId,
) -> Result<TextLayoutPreviewMetrics> {
    let element_acceptance = acceptance
        .pages
        .iter()
        .flat_map(|page| &page.elements)
        .find(|value| value.element_id == element)
        .with_context(|| format!("layout measurements are unavailable for {element}"))?;
    let measurements = &element_acceptance.measurements;
    Ok(TextLayoutPreviewMetrics {
        deterministic_element_accepted: element_acceptance.accepted,
        deterministic_rejection_codes: acceptance
            .rejection_reasons
            .iter()
            .filter(|rejection| {
                rejection.element_id == Some(element)
                    || rejection.related_element_id == Some(element)
            })
            .map(|rejection| rejection_code_name(rejection.code))
            .collect(),
        finite_positive_layout: measurements.finite_positive_layout,
        line_count: measurements.rendered_line_count,
        overflow: measurements
            .renderer_diagnostics
            .iter()
            .any(|diagnostic| diagnostic == "text_overflow"),
        rendered_font_size_px: measurements.rendered_font_size_px,
        rendered_glyph_height_px: measurements.rendered_glyph_height_px,
        maximum_reasonable_line_count: measurements.maximum_reasonable_line_count,
        target_layout_anchor: Some(measurements.target_layout_anchor.clone()),
        page_overflow_px: measurements.page_overflow_px,
        text_safe_clearance: measurements.text_safe_containment.clone(),
        renderer_diagnostics: measurements.renderer_diagnostics.clone(),
    })
}

fn deterministic_diagnostics_for(
    acceptance: &AcceptanceRecord,
    element: EntityId,
) -> Vec<AcceptanceRejection> {
    acceptance
        .rejection_reasons
        .iter()
        .filter(|rejection| {
            rejection.element_id == Some(element) || rejection.related_element_id == Some(element)
        })
        .cloned()
        .collect()
}

fn validate_text_layout_candidate(
    before: &TextLayoutPreviewMetrics,
    candidate: &TextLayoutPreviewMetrics,
    max_lines: u8,
    thresholds: &QualityThresholds,
) -> Result<()> {
    if !candidate.deterministic_element_accepted
        || !candidate.deterministic_rejection_codes.is_empty()
    {
        bail!(
            "text layout candidate still fails configured deterministic constraints {} and was not committed",
            serde_json::to_string(&candidate.deterministic_rejection_codes)?
        );
    }
    let line_count = candidate
        .line_count
        .context("text layout candidate line count is unavailable")?;
    if line_count > usize::from(max_lines) {
        bail!(
            "text layout candidate has {line_count} lines, exceeding max_lines {max_lines}; no mutation made"
        );
    }
    if candidate
        .page_overflow_px
        .is_some_and(|value| value > thresholds.max_page_overflow_px)
    {
        bail!("text layout candidate exceeds page overflow threshold; no mutation made");
    }
    if !text_layout_metrics_improved(before, candidate) {
        bail!(
            "text layout candidate does not strictly improve the blocking deterministic layout result; no mutation made"
        );
    }
    Ok(())
}

fn validate_compact_translation_candidate(
    before: &TextLayoutPreviewMetrics,
    candidate: &TextLayoutPreviewMetrics,
    thresholds: &QualityThresholds,
) -> Result<()> {
    if !candidate.deterministic_element_accepted
        || !candidate.deterministic_rejection_codes.is_empty()
    {
        bail!(
            "compact translation candidate still fails configured deterministic constraints {} and was not committed",
            serde_json::to_string(&candidate.deterministic_rejection_codes)?
        );
    }
    if candidate
        .page_overflow_px
        .is_none_or(|value| value > thresholds.max_page_overflow_px)
    {
        bail!("compact translation candidate exceeds page overflow threshold; no mutation made");
    }
    if !text_layout_metrics_improved(before, candidate) {
        bail!(
            "compact translation candidate is safe but does not strictly improve layout metrics; no mutation made"
        );
    }
    Ok(())
}

fn text_layout_metrics_improved(
    before: &TextLayoutPreviewMetrics,
    candidate: &TextLayoutPreviewMetrics,
) -> bool {
    (!before.deterministic_element_accepted && candidate.deterministic_element_accepted)
        || candidate.deterministic_rejection_codes.len()
            < before.deterministic_rejection_codes.len()
        || (before.overflow && !candidate.overflow)
        || optional_metric_decreased(
            before
                .target_layout_anchor
                .as_ref()
                .and_then(|anchor| anchor.rendered_overflow_px),
            candidate
                .target_layout_anchor
                .as_ref()
                .and_then(|anchor| anchor.rendered_overflow_px),
        )
        || optional_metric_decreased(before.page_overflow_px, candidate.page_overflow_px)
        || optional_metric_decreased(
            before.line_count.map(|value| value as f64),
            candidate.line_count.map(|value| value as f64),
        )
        || optional_metric_increased(
            before
                .text_safe_clearance
                .as_ref()
                .and_then(|value| value.minimum_clearance_px),
            candidate
                .text_safe_clearance
                .as_ref()
                .and_then(|value| value.minimum_clearance_px),
        )
}

fn optional_metric_decreased(before: Option<f64>, candidate: Option<f64>) -> bool {
    matches!((before, candidate), (Some(before), Some(candidate)) if candidate + f64::EPSILON < before)
}

fn optional_metric_increased(before: Option<f64>, candidate: Option<f64>) -> bool {
    matches!((before, candidate), (Some(before), Some(candidate)) if candidate > before + f64::EPSILON)
}

async fn commit_text_layout_patch_if_safe(
    session: &mut Session,
    patch: Patch,
    before: &TextLayoutPreviewMetrics,
    candidate: &TextLayoutPreviewMetrics,
    max_lines: u8,
    thresholds: &QualityThresholds,
) -> Result<Revision> {
    validate_text_layout_candidate(before, candidate, max_lines, thresholds)?;
    Ok(session.commit(patch).await?.snapshot.revision())
}

async fn commit_compact_translation_patch_if_safe(
    session: &mut Session,
    patch: Patch,
    preview_base_revision: Revision,
    before: &TextLayoutPreviewMetrics,
    candidate: &TextLayoutPreviewMetrics,
    thresholds: &QualityThresholds,
) -> Result<Revision> {
    if session.snapshot().revision() != preview_base_revision {
        bail!(
            "compact translation preview is stale: previewed revision {preview_base_revision}, current revision {}",
            session.snapshot().revision()
        );
    }
    validate_compact_translation_candidate(before, candidate, thresholds)?;
    Ok(session.commit(patch).await?.snapshot.revision())
}

fn validate_text_safe_inset_delta(delta: TextSafeInsetDelta) -> Result<()> {
    for (edge, value) in [
        ("top", delta.top),
        ("right", delta.right),
        ("bottom", delta.bottom),
        ("left", delta.left),
    ] {
        if !value.is_finite() || value <= 0.0 {
            bail!("text-safe inset delta {edge} must be finite and greater than zero");
        }
    }
    Ok(())
}

fn increased_text_safe_padding_geometry(
    snapshot: &Snapshot,
    element: EntityId,
    content: EntityId,
    delta: TextSafeInsetDelta,
) -> Result<(Geometry, ElementBounds, ElementBounds)> {
    validate_text_safe_inset_delta(delta)?;
    let layer = snapshot.text_layer(element)?;
    let content = snapshot.text_content(content)?;
    let source_region = content.source_region()?;
    let safe_region = match layer.balloon_target()? {
        Some(region) => Some(region),
        None => source_region,
    }
    .context("text-safe layout repair requires a detected balloon or text-safe source region")?;
    let fallback_layout_bounds = source_region.unwrap_or(safe_region).geometry()?;
    let before = snapshot
        .component::<Geometry>(element)?
        .map(element_geometry)
        .map_or_else(
            || element_geometry(fallback_layout_bounds).bounds,
            |geometry| geometry.bounds,
        );
    if !valid_bounds(before) {
        bail!("current text layout bounds must be finite and positive");
    }
    let candidate = inset_layout_bounds(before, delta)?;
    Ok((
        Geometry::rectangle(candidate.x, candidate.y, candidate.width, candidate.height),
        before,
        candidate,
    ))
}

fn inset_layout_bounds(before: ElementBounds, delta: TextSafeInsetDelta) -> Result<ElementBounds> {
    validate_text_safe_inset_delta(delta)?;
    if !valid_bounds(before) {
        bail!("current text layout bounds must be finite and positive");
    }
    let candidate = ElementBounds {
        x: before.x + delta.left,
        y: before.y + delta.top,
        width: before.width - delta.left - delta.right,
        height: before.height - delta.top - delta.bottom,
    };
    if !valid_bounds(candidate) {
        bail!("text-safe inset delta leaves no positive text layout interior");
    }
    validate_nonexpanding_layout_bounds(before, candidate)?;
    Ok(candidate)
}

fn validate_nonexpanding_layout_bounds(
    before: ElementBounds,
    candidate: ElementBounds,
) -> Result<()> {
    let before_right = before.x + before.width;
    let before_bottom = before.y + before.height;
    let candidate_right = candidate.x + candidate.width;
    let candidate_bottom = candidate.y + candidate.height;
    if candidate.x <= before.x
        || candidate.y <= before.y
        || candidate_right >= before_right
        || candidate_bottom >= before_bottom
    {
        bail!("text-safe layout repair must strictly inset every edge and cannot expand geometry");
    }
    Ok(())
}

fn text_safe_containment_for(
    acceptance: &AcceptanceRecord,
    element: EntityId,
) -> Result<&TextSafeContainment> {
    acceptance
        .pages
        .iter()
        .flat_map(|page| &page.elements)
        .find(|value| value.element_id == element)
        .and_then(|value| value.measurements.text_safe_containment.as_ref())
        .with_context(|| format!("text-safe containment measurement is unavailable for {element}"))
}

fn validate_text_safe_clearance_improvement(
    before: &TextSafeContainment,
    candidate: &TextSafeContainment,
) -> Result<()> {
    let before_clearance = before
        .minimum_clearance_px
        .filter(|value| value.is_finite())
        .context("current minimum text-safe clearance is unavailable or non-finite")?;
    let candidate_clearance = candidate
        .minimum_clearance_px
        .filter(|value| value.is_finite())
        .context("candidate minimum text-safe clearance is unavailable or non-finite")?;
    if before.region_id != candidate.region_id {
        bail!("text-safe region changed while previewing the layout repair");
    }
    if candidate.required_padding_px != before.required_padding_px {
        bail!("text-safe clearance threshold changed while previewing the layout repair");
    }
    if candidate_clearance <= before_clearance {
        bail!(
            "text-safe layout repair rejected without mutation: minimum clearance must strictly increase (before {before_clearance:.2}px, candidate {candidate_clearance:.2}px)"
        );
    }
    if candidate_clearance < candidate.required_padding_px {
        bail!(
            "text-safe layout repair rejected without mutation: candidate clearance {candidate_clearance:.2}px does not reach required {:.2}px",
            candidate.required_padding_px
        );
    }
    Ok(())
}

fn validate_text_safe_layout_candidate(
    before: &TextLayoutPreviewMetrics,
    candidate: &TextLayoutPreviewMetrics,
    thresholds: &QualityThresholds,
) -> Result<()> {
    if !candidate.deterministic_element_accepted
        || !candidate.deterministic_rejection_codes.is_empty()
    {
        bail!(
            "text-safe layout repair rejected without mutation: candidate fails per-element deterministic constraints {}",
            serde_json::to_string(&candidate.deterministic_rejection_codes)?
        );
    }
    if !candidate.finite_positive_layout || !candidate.renderer_diagnostics.is_empty() {
        bail!(
            "text-safe layout repair rejected without mutation: candidate layout is non-finite, clipped, or has renderer diagnostics"
        );
    }
    let line_count = candidate
        .line_count
        .context("candidate rendered line count is unavailable")?;
    if line_count == 0 || line_count > candidate.maximum_reasonable_line_count {
        bail!(
            "text-safe layout repair rejected without mutation: candidate line count {line_count} is outside 1..={} ",
            candidate.maximum_reasonable_line_count
        );
    }
    let font_size = candidate
        .rendered_font_size_px
        .context("candidate rendered font size is unavailable")?;
    if font_size < thresholds.min_rendered_font_size_px {
        bail!(
            "text-safe layout repair rejected without mutation: rendered_font_size_below_minimum ({font_size:.6}px < {:.2}px)",
            thresholds.min_rendered_font_size_px
        );
    }
    let glyph_height = candidate
        .rendered_glyph_height_px
        .context("candidate rendered glyph height is unavailable")?;
    if glyph_height < thresholds.min_rendered_glyph_height_px {
        bail!(
            "text-safe layout repair rejected without mutation: rendered_glyph_height_below_minimum ({glyph_height:.6}px < {:.2}px)",
            thresholds.min_rendered_glyph_height_px
        );
    }
    let anchor = candidate
        .target_layout_anchor
        .as_ref()
        .context("candidate target/source layout anchor is unavailable")?;
    if anchor
        .rendered_area_coverage
        .zip(anchor.required_area_coverage)
        .is_none_or(|(actual, required)| actual < required)
    {
        bail!(
            "text-safe layout repair rejected without mutation: candidate fails target/source anchor density coverage"
        );
    }
    if anchor
        .rendered_overflow_px
        .is_none_or(|overflow| overflow > thresholds.max_layout_anchor_overflow_px)
    {
        bail!(
            "text-safe layout repair rejected without mutation: candidate fails target/source anchor containment"
        );
    }
    if candidate
        .page_overflow_px
        .is_none_or(|overflow| overflow > thresholds.max_page_overflow_px)
    {
        bail!(
            "text-safe layout repair rejected without mutation: candidate fails page-overflow containment"
        );
    }
    let before_clearance = before
        .text_safe_clearance
        .as_ref()
        .context("current text-safe clearance measurement is unavailable")?;
    let candidate_clearance = candidate
        .text_safe_clearance
        .as_ref()
        .context("candidate text-safe clearance measurement is unavailable")?;
    if candidate_clearance.required_padding_px != thresholds.min_text_safe_padding_px
        || candidate_clearance.violation_pixels != 0
    {
        bail!(
            "text-safe layout repair rejected without mutation: candidate fails configured {:.2}px contour clearance",
            thresholds.min_text_safe_padding_px
        );
    }
    validate_text_safe_clearance_improvement(before_clearance, candidate_clearance)
}

async fn commit_text_safe_patch_if_globally_safe(
    session: &mut Session,
    patch: Patch,
    preview_base_revision: Revision,
    before: &TextLayoutPreviewMetrics,
    candidate: &TextLayoutPreviewMetrics,
    thresholds: &QualityThresholds,
) -> Result<Revision> {
    if session.snapshot().revision() != preview_base_revision {
        bail!(
            "text-safe repair preview is stale: previewed revision {preview_base_revision}, current revision {}",
            session.snapshot().revision()
        );
    }
    validate_text_safe_layout_candidate(before, candidate, thresholds)?;
    Ok(session.commit(patch).await?.snapshot.revision())
}

fn review_requires_layout_repair(
    review: &VisualReviewRecord,
    acceptance: Option<&AcceptanceRecord>,
    element: EntityId,
) -> bool {
    let command_layout_rejection = review
        .decision
        .as_ref()
        .is_some_and(|decision| !decision.judgments.typography_layout_acceptable);
    let agent_layout_rejection = review
        .agent_reviews
        .iter()
        .any(|submitted| !submitted.decision.judgments.typography_layout_acceptable);
    command_layout_rejection
        || agent_layout_rejection
        || acceptance.is_some_and(|record| {
            record.rejection_reasons.iter().any(|rejection| {
                (rejection.element_id == Some(element)
                    || rejection.related_element_id == Some(element)
                    || rejection.element_id.is_none())
                    && matches!(
                        rejection.code,
                        AcceptanceRejectionCode::TypesettingFailed
                            | AcceptanceRejectionCode::TranslationNotRenderEligible
                            | AcceptanceRejectionCode::TranslationGeometryNotVisible
                            | AcceptanceRejectionCode::UnsupportedLayout
                            | AcceptanceRejectionCode::RenderedFontSizeMissing
                            | AcceptanceRejectionCode::RenderedFontSizeBelowMinimum
                            | AcceptanceRejectionCode::RenderedLineCountMissing
                            | AcceptanceRejectionCode::RenderedLineCountUnreasonable
                            | AcceptanceRejectionCode::RenderedGlyphHeightBelowMinimum
                            | AcceptanceRejectionCode::SourceRegionCoverageBelowMinimum
                            | AcceptanceRejectionCode::RenderedTextOutsideSourceRegion
                            | AcceptanceRejectionCode::TargetAnchorCoverageBelowMinimum
                            | AcceptanceRejectionCode::RenderedTextOutsideTargetAnchor
                            | AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior
                            | AcceptanceRejectionCode::RenderedTextOutsidePage
                            | AcceptanceRejectionCode::TranslatedRegionOverlap
                            | AcceptanceRejectionCode::SourceRegionOverlap
                    )
            })
        })
}

fn preserve_original_ocr(
    previous_original: Option<RepairText>,
    current_source: Option<RepairText>,
) -> Option<RepairText> {
    previous_original.or(current_source)
}

fn update_stage_telemetry(
    stages: &SyncMutex<BTreeMap<(EntityId, Stage), MutableStageTelemetry>>,
    event: Progress,
) {
    let (page, stage, status, model, elapsed_ms) = match event {
        Progress::Started { .. } => return,
        Progress::Loading { page, stage, model } => (
            page,
            stage,
            StageTelemetryStatus::Loading,
            Some(model),
            None,
        ),
        Progress::Running { page, stage, model } => (
            page,
            stage,
            StageTelemetryStatus::Running,
            Some(model),
            None,
        ),
        Progress::Finished {
            page,
            stage,
            model,
            elapsed,
        } => (
            page,
            stage,
            StageTelemetryStatus::Finished,
            Some(model),
            Some(elapsed.as_millis()),
        ),
        Progress::NoOp {
            page,
            stage,
            model,
            elapsed,
        } => (
            page,
            stage,
            StageTelemetryStatus::NoOp,
            Some(model),
            Some(elapsed.as_millis()),
        ),
        Progress::Skipped { page, stage } => {
            (page, stage, StageTelemetryStatus::Skipped, None, None)
        }
        Progress::Preprocessed { .. } => return,
    };
    let mut stages = stages.lock();
    let entry = stages
        .entry((page, stage))
        .or_insert(MutableStageTelemetry {
            status: StageTelemetryStatus::NotRun,
            model: None,
            elapsed_ms: None,
        });
    entry.status = status;
    if model.is_some() {
        entry.model = model;
    }
    if elapsed_ms.is_some() {
        entry.elapsed_ms = elapsed_ms;
    }
}

fn build_pipeline_telemetry(
    status: PipelineRunTelemetryStatus,
    failure_stage: Option<Stage>,
    failure_kind: Option<String>,
    failure: Option<String>,
    page_labels: &BTreeMap<EntityId, String>,
    stages: &BTreeMap<(EntityId, Stage), MutableStageTelemetry>,
    inspection: &ProjectInspection,
    target_language: &str,
) -> PipelineTelemetry {
    let pages = page_labels
        .iter()
        .map(|(page_id, label)| {
            let page = inspection
                .project
                .pages
                .iter()
                .find(|page| page.id == *page_id);
            let semantic_after = page.map_or_else(StageSemanticCounts::default, |page| {
                let mut counts = StageSemanticCounts::default();
                counts.logical_dialogue_groups = page.logical_dialogue_groups.len();
                for element in &page.text_elements {
                    counts.detected_text_elements += usize::from(element.detected);
                    if element
                        .decorative_sfx
                        .as_ref()
                        .is_some_and(DecorativeSfxDecision::is_skipped)
                    {
                        counts.skipped_difficult_sfx += 1;
                    }
                    if !element.required {
                        continue;
                    }
                    counts.required_source_elements += 1;
                    let render_owner =
                        element
                            .logical_dialogue_memberships
                            .first()
                            .is_none_or(|membership| {
                                membership.primary_render_element_id == element.id
                            });
                    counts.grouped_source_members +=
                        usize::from(!element.logical_dialogue_memberships.is_empty());
                    counts.required_render_units += usize::from(render_owner);
                    if let Some(source) = &element.source {
                        counts.source_text_present += 1;
                        counts.source_text_nonempty += usize::from(!source.text.trim().is_empty());
                    }
                    if render_owner && let Some(translation) = &element.translation {
                        counts.translation_present += 1;
                        counts.translation_nonempty +=
                            usize::from(!translation.text.trim().is_empty());
                        counts.target_language_translations +=
                            usize::from(translation.language.as_deref() == Some(target_language));
                    }
                    if render_owner {
                        counts.render_eligible_translations +=
                            usize::from(element.final_scene.eligible);
                        counts.visible_translations += usize::from(element.final_scene.visible);
                    }
                }
                counts
            });
            let stages =
                Stage::ALL
                    .into_iter()
                    .map(|stage| {
                        let value = stages.get(&(*page_id, stage)).cloned().unwrap_or(
                            MutableStageTelemetry {
                                status: StageTelemetryStatus::NotRun,
                                model: None,
                                elapsed_ms: None,
                            },
                        );
                        StageTelemetry {
                            stage,
                            status: value.status,
                            model: value.model,
                            elapsed_ms: value.elapsed_ms,
                        }
                    })
                    .collect::<Vec<_>>();
            let stage_status = |stage| {
                stages
                    .iter()
                    .find(|value| value.stage == stage)
                    .map(|value| value.status)
                    .unwrap_or(StageTelemetryStatus::NotRun)
            };
            let mut diagnosis = Vec::new();
            if semantic_after.detected_text_elements == 0 {
                diagnosis.push(PipelineDiagnosis::NoDetection);
            } else if semantic_after.source_text_nonempty < semantic_after.required_source_elements
            {
                diagnosis.push(PipelineDiagnosis::OcrEmpty);
            }
            match stage_status(Stage::Translation) {
                StageTelemetryStatus::Skipped | StageTelemetryStatus::NoOp
                    if semantic_after.source_text_nonempty > 0 =>
                {
                    diagnosis.push(PipelineDiagnosis::TranslationSkipped);
                }
                StageTelemetryStatus::Failed => {
                    diagnosis.push(PipelineDiagnosis::TranslationFailed);
                }
                StageTelemetryStatus::Finished
                    if semantic_after.target_language_translations
                        < semantic_after.required_render_units =>
                {
                    diagnosis.push(PipelineDiagnosis::TranslationOutputIncomplete);
                }
                _ => {}
            }
            if page.is_some_and(|page| page.render_error.is_some())
                || semantic_after.visible_translations < semantic_after.target_language_translations
            {
                diagnosis.push(PipelineDiagnosis::TypesettingFailed);
            }
            if diagnosis.is_empty()
                && semantic_after.visible_translations == semantic_after.required_render_units
                && semantic_after.source_text_nonempty == semantic_after.required_source_elements
                && semantic_after.detected_text_elements > 0
            {
                diagnosis.push(PipelineDiagnosis::ReadyForExport);
            }
            PagePipelineTelemetry {
                page_id: *page_id,
                label: label.clone(),
                stages,
                semantic_after,
                diagnosis,
            }
        })
        .collect();
    PipelineTelemetry {
        schema_version: 2,
        status,
        pages,
        failure_stage,
        failure_kind,
        failure,
    }
}

fn validate_required_translations(pages: &[PagePipelineTelemetry]) -> Result<()> {
    let mut failures = Vec::new();
    for page in pages {
        let counts = &page.semantic_after;
        if counts.source_text_nonempty != counts.required_source_elements {
            failures.push(format!(
                "page {} ({}) has {} nonempty source texts for {} required source members",
                page.page_id,
                page.label,
                counts.source_text_nonempty,
                counts.required_source_elements,
            ));
        }
        if counts.translation_present != counts.required_render_units
            || counts.translation_nonempty != counts.required_render_units
            || counts.target_language_translations != counts.required_render_units
        {
            failures.push(format!(
                "page {} ({}) has {}/{}/{} present, nonempty, target-language translations for {} required render units",
                page.page_id,
                page.label,
                counts.translation_present,
                counts.translation_nonempty,
                counts.target_language_translations,
                counts.required_render_units,
            ));
        }
    }
    ensure!(failures.is_empty(), "{}", failures.join("; "));
    Ok(())
}

fn mark_pipeline_completed(completed: &AtomicBool, pages: &[PagePipelineTelemetry]) -> Result<()> {
    validate_required_translations(pages)?;
    completed.store(true, Ordering::Release);
    Ok(())
}

#[derive(Deserialize, JsonSchema)]
struct InspectProject {}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ViewPage {
    page_ordinal: PageOrdinal,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct InspectPageEvidence {
    page_ordinal: PageOrdinal,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ReviewPageTranslation {
    page_ordinal: PageOrdinal,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct InspectSourceEvidence {
    page_ordinal: PageOrdinal,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ViewPageSourceDebug {
    page_ordinal: PageOrdinal,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ViewPageDebug {
    page_ordinal: PageOrdinal,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RunSourceAnalysis {}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ClassifyDecorativeSfx {
    page_ordinal: PageOrdinal,
    scene_revision: u64,
    source_evidence_dossier_blake3: String,
    source_debug_artifact_blake3: String,
    decisions: Vec<DecorativeSfxClassification>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct DecorativeSfxClassification {
    element: String,
    original_ordinal: usize,
    disposition: DecorativeSfxDisposition,
    source_crop_blake3: String,
    source_debug_label: String,
    confidence: f32,
    evidence: DecorativeSfxEvidence,
    rationale: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct VerifyUiPanelAnchor {
    page_ordinal: PageOrdinal,
    scene_revision: u64,
    source_evidence_dossier_blake3: String,
    source_debug_artifact_blake3: String,
    decisions: Vec<UiPanelVerification>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct UiPanelVerification {
    element: String,
    original_ordinal: usize,
    classification: UiPanelClassificationKind,
    source_crop_blake3: String,
    source_debug_label: String,
    panel_region: String,
    confidence: f32,
    evidence: UiPanelEvidence,
    association_reason: String,
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum UiPanelClassificationKind {
    RequiredUiPanelText,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RunPipeline {}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ReviewPages {}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SubmitVisualSemanticReview {
    page_ordinal: PageOrdinal,
    scene_revision: u64,
    source_evidence_dossier_blake3: String,
    source_debug_artifact_blake3: String,
    dossier_blake3: String,
    debug_artifact_blake3: String,
    compacted_translation_reviews: Vec<CompactedTranslationSemanticReview>,
    decision: VisualReviewDecision,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ReviseElement {
    element: String,
    source_text: Option<String>,
    translation_text: Option<String>,
    typography: Option<RepairTypography>,
    layout: Option<RepairLayout>,
    reason: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RevisePageTranslation {
    page_ordinal: PageOrdinal,
    evidence: PageTranslationEvidenceReference,
    edits: Vec<PageTranslationEdit>,
    page_rationale: String,
}

#[derive(Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PageTranslationEvidenceReference {
    scene_revision: u64,
    source_evidence_dossier_blake3: String,
    source_debug_artifact_blake3: String,
    dossier_blake3: String,
    debug_artifact_blake3: String,
}

#[derive(Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PageTranslationEdit {
    element: String,
    source: Option<LanguageTextEdit>,
    translation: Option<LanguageTextEdit>,
}

#[derive(Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct LanguageTextEdit {
    text: String,
    language: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PreviewTextLayout {
    element: String,
    options: TextLayoutOptions,
    reason: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(untagged)]
enum TextLayoutOptions {
    SourceBoundInterjectionFallback(SourceBoundInterjectionFallbackOptions),
    Controlled(ControlledTextLayoutOptions),
}

impl TextLayoutOptions {
    fn line_break_policy(&self) -> KoreanLineBreakPolicy {
        match self {
            Self::SourceBoundInterjectionFallback(_) => KoreanLineBreakPolicy::PreserveExisting,
            Self::Controlled(options) => options.line_break_policy,
        }
    }

    fn max_lines(&self) -> u8 {
        match self {
            Self::SourceBoundInterjectionFallback(_) => 1,
            Self::Controlled(options) => options.max_lines,
        }
    }

    fn safe_padding_increase_px(&self) -> Option<TextSafeInsetDelta> {
        match self {
            Self::SourceBoundInterjectionFallback(_) => None,
            Self::Controlled(options) => options.safe_padding_increase_px,
        }
    }

    fn source_bound_interjection_fallback(&self) -> Option<SourceBoundInterjectionFallback> {
        match self {
            Self::SourceBoundInterjectionFallback(options) => {
                Some(options.source_bound_interjection_fallback)
            }
            Self::Controlled(_) => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
struct ControlledTextLayoutOptions {
    line_break_policy: KoreanLineBreakPolicy,
    max_lines: u8,
    alignment: Option<ControlledTextAlignment>,
    font_scale: Option<f32>,
    safe_padding_increase_px: Option<TextSafeInsetDelta>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceBoundInterjectionFallbackOptions {
    source_bound_interjection_fallback: SourceBoundInterjectionFallback,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum SourceBoundInterjectionFallback {
    NativeVertical,
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(rename_all = "snake_case")]
enum KoreanLineBreakPolicy {
    PreserveExisting,
    KoreanKeepWordsBalanced,
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(rename_all = "snake_case")]
enum ControlledTextAlignment {
    Start,
    Center,
    End,
}

impl From<ControlledTextAlignment> for TextAlignment {
    fn from(value: ControlledTextAlignment) -> Self {
        match value {
            ControlledTextAlignment::Start => Self::Start,
            ControlledTextAlignment::Center => Self::Center,
            ControlledTextAlignment::End => Self::End,
        }
    }
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct CommitTextLayout {
    preview_id: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PreviewCompactTranslation {
    logical_group: String,
    primary_render_element: String,
    candidate_translation: LanguageTextEdit,
    evidence: PageTranslationEvidenceReference,
    rationale: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct CommitCompactTranslation {
    preview_id: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PreviewTextSafeLayoutRepair {
    element: String,
    inset_delta_px: TextSafeInsetDelta,
    reason: String,
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
struct TextSafeInsetDelta {
    top: f64,
    right: f64,
    bottom: f64,
    left: f64,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct CommitTextSafeLayoutRepair {
    preview_id: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RepairTypography {
    preferred_font: Option<String>,
    font_weight: Option<u16>,
    font_style: Option<RepairFontStyle>,
    size: Option<f32>,
    auto_fit: Option<bool>,
    color: Option<[u8; 4]>,
    stroke_color: Option<[u8; 4]>,
    stroke_width: Option<f32>,
    alignment: Option<RepairTextAlignment>,
    writing_mode: Option<RepairWritingMode>,
}

impl RepairTypography {
    fn to_scene(&self, existing: Option<&Typography>) -> Typography {
        Typography {
            origin: koharu_scene::Origin::User,
            preferred_font: self
                .preferred_font
                .clone()
                .or_else(|| existing.and_then(|value| value.preferred_font.clone())),
            font_weight: self
                .font_weight
                .or_else(|| existing.and_then(|value| value.font_weight)),
            font_style: self
                .font_style
                .map(Into::into)
                .or_else(|| existing.and_then(|value| value.font_style)),
            size: self.size.or_else(|| existing.and_then(|value| value.size)),
            auto_fit: self
                .auto_fit
                .unwrap_or_else(|| existing.is_none_or(|value| value.auto_fit)),
            color: self
                .color
                .or_else(|| existing.and_then(|value| value.color)),
            stroke_color: self
                .stroke_color
                .or_else(|| existing.and_then(|value| value.stroke_color)),
            stroke_width: self
                .stroke_width
                .or_else(|| existing.and_then(|value| value.stroke_width)),
            alignment: self
                .alignment
                .map(Into::into)
                .or_else(|| existing.and_then(|value| value.alignment)),
            writing_mode: self
                .writing_mode
                .map(Into::into)
                .or_else(|| existing.and_then(|value| value.writing_mode)),
            extensions: existing
                .map(|typography| typography.extensions.clone())
                .unwrap_or_default(),
        }
    }
}

#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum RepairFontStyle {
    Normal,
    Italic,
    Oblique,
}

impl From<RepairFontStyle> for FontStyle {
    fn from(value: RepairFontStyle) -> Self {
        match value {
            RepairFontStyle::Normal => Self::Normal,
            RepairFontStyle::Italic => Self::Italic,
            RepairFontStyle::Oblique => Self::Oblique,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(rename_all = "snake_case")]
enum RepairTextAlignment {
    Start,
    Center,
    End,
    Justify,
}

impl From<RepairTextAlignment> for TextAlignment {
    fn from(value: RepairTextAlignment) -> Self {
        match value {
            RepairTextAlignment::Start => Self::Start,
            RepairTextAlignment::Center => Self::Center,
            RepairTextAlignment::End => Self::End,
            RepairTextAlignment::Justify => Self::Justify,
        }
    }
}

#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum RepairWritingMode {
    Horizontal,
    Vertical,
}

impl From<RepairWritingMode> for WritingMode {
    fn from(value: RepairWritingMode) -> Self {
        match value {
            RepairWritingMode::Horizontal => Self::Horizontal,
            RepairWritingMode::Vertical => Self::Vertical,
        }
    }
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RepairLayout {
    kind: Option<RepairLayoutKind>,
    bounds: Option<RepairBounds>,
    padding: Option<RepairPadding>,
}

#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RepairPadding {
    top: f64,
    right: f64,
    bottom: f64,
    left: f64,
}

#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum RepairLayoutKind {
    Point,
    Paragraph,
}

impl From<RepairLayoutKind> for TextLayoutKind {
    fn from(value: RepairLayoutKind) -> Self {
        match value {
            RepairLayoutKind::Point => Self::Point,
            RepairLayoutKind::Paragraph => Self::Paragraph,
        }
    }
}

#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RepairBounds {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

impl From<OutputFormat> for ExportFormat {
    fn from(value: OutputFormat) -> Self {
        match value {
            OutputFormat::Png => Self::Png,
            OutputFormat::Psd => Self::Psd,
        }
    }
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ExportPages {}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_provider_image_urls(value: &Value) {
        match value {
            Value::Array(values) => {
                for value in values {
                    assert_provider_image_urls(value);
                }
            }
            Value::Object(object) => {
                if let Some(url) = object.get("image_url").and_then(Value::as_str) {
                    assert!(
                        url.starts_with("data:")
                            || url.starts_with("http://")
                            || url.starts_with("https://"),
                        "unsupported model image URL: {url}"
                    );
                    assert!(
                        !Path::new(url).is_absolute(),
                        "absolute local path used as model image URL: {url}"
                    );
                }
                for value in object.values() {
                    assert_provider_image_urls(value);
                }
            }
            _ => {}
        }
    }

    fn assert_text_json_only(value: &Value) {
        match value {
            Value::Array(values) => {
                for value in values {
                    assert_text_json_only(value);
                }
            }
            Value::Object(object) => {
                assert_ne!(
                    object.get("type").and_then(Value::as_str),
                    Some("input_image")
                );
                assert!(!object.contains_key("image_url"));
                for value in object.values() {
                    assert_text_json_only(value);
                }
            }
            _ => {}
        }
    }

    fn assert_compacted_media_evidence(kind: &str, artifacts: &[(&str, &str)]) {
        let directory = tempfile::tempdir().unwrap();
        let artifact_values = artifacts
            .iter()
            .map(|(path, media_type)| {
                json!({
                    "path": path,
                    "media_type": media_type,
                    "byte_length": 3,
                    "blake3": "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85",
                })
            })
            .collect::<Vec<_>>();
        let mut output = vec![json!({
            "type": "input_text",
            "text": serde_json::to_string(&json!({
                "ok": true,
                "value": {
                    "payload_compacted": true,
                    "payload_kind": kind,
                    "artifacts": artifact_values,
                }
            }))
            .unwrap(),
        })];
        for (index, (_, media_type)) in artifacts.iter().enumerate() {
            output.push(json!({
                "type": "input_text",
                "text": format!(
                    "<koharu_image_artifact>\n{}\n</koharu_image_artifact>",
                    json!({
                        "schema_version": 1,
                        "label": format!("{kind}-{index}"),
                        "provenance": {
                            "kind": "content_hash",
                            "algorithm": "blake3",
                            "digest": "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85",
                            "media_type": media_type,
                            "byte_length": 3,
                        },
                        "image_bytes_retained": false,
                    })
                ),
            }));
            output.push(json!({
                "type": "input_image",
                "image_url": format!(
                    "data:{media_type};base64,{}",
                    "YWJj".repeat(10_000)
                ),
                "detail": "high",
            }));
        }
        let immediate = json!({
            "type": "function_call_output",
            "call_id": format!("call-{kind}"),
            "output": output,
        });
        assert_provider_image_urls(&immediate);

        let mut retained = vec![immediate];
        let released = koharu_agent::release_observed_image_data(&mut retained)
            .unwrap()
            .unwrap();
        assert!(released.1 < released.0);
        assert_text_json_only(&Value::Array(retained.clone()));
        let retained_before_compaction = serde_json::to_string(&retained).unwrap();
        for (path, _) in artifacts {
            assert!(retained_before_compaction.contains(path));
        }

        retained.push(json!({
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": "x".repeat(400_000) }],
        }));
        let compacted = koharu_agent::compact_retained_transcript(directory.path(), &mut retained)
            .unwrap()
            .unwrap();
        assert_text_json_only(&Value::Array(retained));

        let artifact_path = compacted.artifact["path"].as_str().unwrap();
        assert!(Path::new(artifact_path).is_absolute());
        let complete: Value =
            serde_json::from_slice(&std::fs::read(artifact_path).unwrap()).unwrap();
        assert_text_json_only(&complete);
        let complete = serde_json::to_string(&complete).unwrap();
        for (path, _) in artifacts {
            assert!(complete.contains(path));
        }
    }

    #[test]
    fn compacted_source_crop_evidence_retains_text_references_only() {
        assert_compacted_media_evidence(
            "inspect_source_evidence",
            &[
                ("/review/source/full-page.png", "image/png"),
                ("/review/source/crop-1.png", "image/png"),
            ],
        );
    }

    #[test]
    fn compacted_source_debug_evidence_retains_text_references_only() {
        assert_compacted_media_evidence(
            "view_page_source_debug",
            &[("/review/debug/source-overlay.png", "image/png")],
        );
    }

    #[test]
    fn compacted_render_review_evidence_retains_text_references_only() {
        assert_compacted_media_evidence(
            "inspect_page_evidence",
            &[("/review/render/translated-overlay.png", "image/png")],
        );
    }

    fn review(
        status: VisualReviewStatus,
        deterministic_acceptance_passed: bool,
        revision: koharu_scene::Revision,
    ) -> VisualReviewRecord {
        VisualReviewRecord {
            schema_version: crate::review::VISUAL_REVIEW_SCHEMA_VERSION,
            attempt: 1,
            scene_revision: revision,
            deterministic_acceptance_passed,
            deterministic_repair_plan: deterministic_repair_plan(&[], revision, None),
            status,
            bundle: crate::review::ReviewBundle {
                directory: String::new(),
                manifest: String::new(),
                pages: Vec::new(),
            },
            judge: crate::review::ExternalJudge {
                required: true,
                command: None,
                protocol: "test",
            },
            agent_reviews: Vec::new(),
            decision: None,
            error: None,
        }
    }

    #[test]
    fn host_exposes_the_bounded_repair_pipeline_tools() {
        let names = tool_definitions()
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "inspect_project",
                "view_page",
                "run_source_analysis",
                "inspect_source_evidence",
                "view_page_source_debug",
                "classify_decorative_sfx",
                "verify_ui_panel_anchor",
                "inspect_page_evidence",
                "run_pipeline",
                "review_pages",
                "submit_visual_semantic_review",
                "revise_element",
                "revise_page_translation",
                "preview_text_layout",
                "commit_text_layout",
                "preview_compact_translation",
                "commit_compact_translation",
                "preview_increase_text_safe_padding",
                "commit_text_safe_layout_repair",
                "export_pages"
            ]
        );
    }

    #[test]
    fn every_page_targeted_tool_exposes_only_a_one_based_page_ordinal() {
        for name in [
            "view_page",
            "inspect_source_evidence",
            "view_page_source_debug",
            "classify_decorative_sfx",
            "verify_ui_panel_anchor",
            "inspect_page_evidence",
            "submit_visual_semantic_review",
            "revise_page_translation",
        ] {
            let parameters = &tool_definition(name).parameters;
            assert_eq!(parameters["additionalProperties"], false, "{name}");
            assert!(parameters["properties"].get("page").is_none(), "{name}");
            assert_eq!(
                parameters["properties"]["page_ordinal"]["$ref"], "#/$defs/PageOrdinal",
                "{name}"
            );
            assert!(
                parameters["required"]
                    .as_array()
                    .unwrap()
                    .contains(&json!("page_ordinal")),
                "{name}"
            );
            assert_eq!(parameters["$defs"]["PageOrdinal"]["type"], "integer");
            assert_eq!(parameters["$defs"]["PageOrdinal"]["minimum"], 1);
        }
        assert!(
            serde_json::from_value::<InspectSourceEvidence>(json!({ "page_ordinal": 7 })).is_ok()
        );
        assert!(serde_json::from_value::<InspectSourceEvidence>(json!({ "page": "7" })).is_err());
        assert!(
            serde_json::from_value::<InspectSourceEvidence>(json!({ "page_ordinal": "7" }))
                .is_err()
        );
        assert!(
            serde_json::from_value::<InspectSourceEvidence>(json!({ "page_ordinal": 0 })).is_err()
        );
    }

    #[test]
    fn every_production_repair_next_action_has_an_invoke_dispatch_arm() {
        fn literal_tool_names(source: &str, marker: &str) -> BTreeSet<String> {
            source
                .lines()
                .filter_map(|line| {
                    line.split_once(marker)
                        .and_then(|(_, suffix)| suffix.split('"').next())
                        .filter(|name| !name.is_empty())
                        .map(str::to_owned)
                })
                .collect()
        }

        let host_source = include_str!("host.rs")
            .split("#[cfg(test)]\nmod tests")
            .next()
            .unwrap();
        let repair_source = include_str!("repair.rs")
            .split("#[cfg(test)]\nmod tests")
            .next()
            .unwrap();
        let mut emitted = literal_tool_names(host_source, "\"tool\": \"");
        emitted.extend(literal_tool_names(repair_source, "tool: \""));
        assert!(emitted.contains("commit_compact_translation"));

        let invoke_dispatch = host_source
            .split("async fn invoke(&self, call: ToolCall, control: &Control)")
            .nth(1)
            .unwrap()
            .split("fn take_trace_records")
            .next()
            .unwrap();
        for tool in emitted {
            assert!(
                invoke_dispatch.contains(&format!("\"{tool}\" =>")),
                "repair next_action emits {tool}, but HarnessHost::invoke has no dispatch arm"
            );
        }
    }

    fn surface_inputs() -> WorkflowSurfaceInputs {
        WorkflowSurfaceInputs {
            source_analysis_completed: false,
            pipeline_completed: false,
            source_evidence_ready: false,
            decorative_sfx_dispositions_complete: false,
            ui_panel_evidence_available: false,
            page_evidence_revision: None,
            page_evidence_is_current: false,
            current_page_review_accepted: None,
            all_page_reviews_current: false,
            semantic_review_stage: None,
            review: None,
            repair_tool: None,
            repair_stopped: false,
            exported: false,
        }
    }

    fn names(inputs: &WorkflowSurfaceInputs) -> (WorkflowPhase, Vec<&'static str>) {
        phase_tool_names(inputs)
    }

    fn add_bundle_page(record: &mut VisualReviewRecord) {
        let artifact = crate::review::ReviewArtifact {
            path: "/review/artifact".to_owned(),
            media_type: "application/json".to_owned(),
            byte_length: 1,
            blake3: "digest".to_owned(),
        };
        record.bundle.pages.push(crate::review::ReviewBundlePage {
            page_id: EntityId::new(),
            label: "page.png".to_owned(),
            original: artifact.clone(),
            rendered_preview: artifact.clone(),
            semantic_elements: artifact,
        });
    }

    #[test]
    fn phase_surfaces_progress_to_export_without_premature_raw_or_dangerous_tools() {
        let revision = Revision::new(7);
        let mut inputs = surface_inputs();

        let (phase, exposed) = names(&inputs);
        assert_eq!(phase, WorkflowPhase::SourceAnalysis);
        assert_eq!(exposed, ["run_source_analysis"]);
        assert!(!exposed.contains(&"inspect_project"));
        assert!(!exposed.contains(&"revise_element"));
        assert!(!exposed.contains(&"export_pages"));

        inputs.source_analysis_completed = true;
        let (phase, exposed) = names(&inputs);
        assert_eq!(phase, WorkflowPhase::SourceEvidence);
        assert!(exposed.contains(&"inspect_source_evidence"));
        assert!(exposed.contains(&"view_page_source_debug"));
        assert!(!exposed.contains(&"inspect_project"));
        assert!(!exposed.contains(&"run_pipeline"));

        inputs.source_evidence_ready = true;
        inputs.decorative_sfx_dispositions_complete = true;
        inputs.ui_panel_evidence_available = true;
        let (_, exposed) = names(&inputs);
        assert!(exposed.contains(&"classify_decorative_sfx"));
        assert!(exposed.contains(&"verify_ui_panel_anchor"));
        assert!(exposed.contains(&"run_pipeline"));

        inputs.pipeline_completed = true;
        inputs.source_evidence_ready = false;
        inputs.ui_panel_evidence_available = false;
        let (phase, exposed) = names(&inputs);
        assert_eq!(phase, WorkflowPhase::DeterministicReview);
        assert!(!exposed.contains(&"inspect_project"));
        assert!(exposed.contains(&"inspect_page_evidence"));
        assert!(exposed.contains(&"review_pages"));
        assert!(!exposed.contains(&"revise_page_translation"));

        inputs.page_evidence_revision = Some(revision);
        inputs.page_evidence_is_current = true;
        let (phase, exposed) = names(&inputs);
        assert_eq!(phase, WorkflowPhase::DeterministicReview);
        assert!(!exposed.contains(&"inspect_project"));
        assert!(exposed.contains(&"review_pages"));
        assert!(exposed.contains(&"revise_page_translation"));
        assert!(!exposed.contains(&"revise_element"));

        let mut pending = review(VisualReviewStatus::PendingAgentReview, true, revision);
        add_bundle_page(&mut pending);
        inputs.review = Some(pending.clone());
        inputs.semantic_review_stage = Some(SemanticReviewStage::SubmitVisualSemanticReview);
        let (phase, exposed) = names(&inputs);
        assert_eq!(phase, WorkflowPhase::SemanticReview);
        assert!(!exposed.contains(&"inspect_project"));
        assert!(exposed.contains(&"submit_visual_semantic_review"));
        assert!(!exposed.contains(&"export_pages"));
        assert!(!exposed.iter().any(|name| name.starts_with("preview_")));

        pending.status = VisualReviewStatus::Accepted;
        inputs.review = Some(pending);
        inputs.all_page_reviews_current = true;
        let (phase, exposed) = names(&inputs);
        assert_eq!(phase, WorkflowPhase::Export);
        assert_eq!(exposed, ["export_pages"]);

        inputs.exported = true;
        let (phase, exposed) = names(&inputs);
        assert_eq!(phase, WorkflowPhase::Completed);
        assert!(exposed.is_empty());
    }

    #[test]
    fn semantic_tool_surface_follows_the_current_evidence_page_review() {
        let revision = Revision::new(7);
        let page = EntityId::new();
        let mut inputs = surface_inputs();
        inputs.source_analysis_completed = true;
        inputs.pipeline_completed = true;
        inputs.page_evidence_revision = Some(revision);
        inputs.page_evidence_is_current = true;
        inputs.review = Some(pending_agent_review(revision, page));
        inputs.semantic_review_stage = Some(SemanticReviewStage::SubmitVisualSemanticReview);

        let (phase, exposed) = names(&inputs);
        assert_eq!(phase, WorkflowPhase::SemanticReview);
        assert!(exposed.contains(&"submit_visual_semantic_review"));

        inputs.current_page_review_accepted = Some(true);
        inputs.semantic_review_stage = Some(SemanticReviewStage::InspectPageEvidence);
        let (phase, exposed) = names(&inputs);
        assert_eq!(phase, WorkflowPhase::PageEvidence);
        assert!(!exposed.contains(&"submit_visual_semantic_review"));

        inputs.current_page_review_accepted = Some(false);
        inputs.review.as_mut().unwrap().status = VisualReviewStatus::Rejected;
        let (phase, exposed) = names(&inputs);
        assert_eq!(phase, WorkflowPhase::SemanticRevision);
        assert!(exposed.contains(&"revise_page_translation"));
    }

    #[test]
    fn completion_policy_continues_only_actionable_repair_phases() {
        let revision = Revision::new(7);
        let mut inputs = surface_inputs();
        inputs.source_analysis_completed = true;
        inputs.pipeline_completed = true;
        inputs.page_evidence_revision = Some(revision);
        inputs.page_evidence_is_current = true;
        inputs.current_page_review_accepted = Some(false);
        let mut rejected = pending_agent_review(revision, EntityId::new());
        rejected.status = VisualReviewStatus::Rejected;
        inputs.review = Some(rejected);

        let (phase, exposed) = names(&inputs);
        let tools = exposed
            .iter()
            .map(|name| tool_definition(name).clone())
            .collect::<Vec<_>>();
        assert_eq!(phase, WorkflowPhase::SemanticRevision);
        assert_eq!(
            actionable_completion(phase, &tools, "revision:7".to_owned(), || {
                Ok("page review rejected inaccurate wording".to_owned())
            })
            .unwrap(),
            Some(HostCompletion::Continue {
                phase: "semantic_revision".to_owned(),
                exposed_tools: exposed.iter().map(|name| (*name).to_owned()).collect(),
                reason: "page review rejected inaccurate wording".to_owned(),
                progress_marker: "revision:7".to_owned(),
            })
        );

        let deterministic_tools = vec![tool_definition("revise_page_translation").clone()];
        assert!(matches!(
            actionable_completion(
                WorkflowPhase::DeterministicRepair,
                &deterministic_tools,
                "revision:7".to_owned(),
                || Ok("active deterministic repair".to_owned()),
            )
            .unwrap(),
            Some(HostCompletion::Continue { .. })
        ));

        inputs.current_page_review_accepted = Some(true);
        inputs.all_page_reviews_current = true;
        inputs.review.as_mut().unwrap().status = VisualReviewStatus::Accepted;
        let (phase, exposed) = names(&inputs);
        let tools = exposed
            .iter()
            .map(|name| tool_definition(name).clone())
            .collect::<Vec<_>>();
        assert_eq!(phase, WorkflowPhase::Export);
        assert_eq!(exposed, ["export_pages"]);
        assert_eq!(
            actionable_completion(phase, &tools, "revision:7".to_owned(), || {
                Ok("already accepted".to_owned())
            })
            .unwrap(),
            None
        );
    }

    #[test]
    fn empty_sfx_attempt_keeps_pipeline_gated_until_dispositions_are_complete() {
        let page = sfx_classification_page(FREE_TEXT_ROLE, "ド");
        let (evidence, mut arguments) = sfx_classification_fixture(&page);
        arguments.decisions.clear();
        assert!(
            validate_decorative_sfx_classifications(
                &page,
                &evidence,
                Revision::new(arguments.scene_revision),
                &arguments,
            )
            .err()
            .expect("empty classification should fail")
            .to_string()
            .contains("requires at least one classification decision")
        );

        let mut inputs = surface_inputs();
        inputs.source_analysis_completed = true;
        inputs.source_evidence_ready = true;

        let (phase, exposed) = names(&inputs);

        assert_eq!(phase, WorkflowPhase::SourceEvidence);
        assert!(exposed.contains(&"classify_decorative_sfx"));
        assert!(!exposed.contains(&"run_pipeline"));
    }

    #[test]
    fn huge_scene_and_raster_payload_becomes_a_bounded_reference_envelope() {
        let directory = tempfile::tempdir().unwrap();
        let elements = (1..=24)
            .map(|ordinal| {
                json!({
                    "ordinal": ordinal,
                    "element_id": format!("element-{ordinal}"),
                    "content_id": format!("content-{ordinal}"),
                    "text_role": "dev.koharu.text.dialogue",
                    "required": true,
                    "current_source": { "text": format!("source {ordinal}"), "language": "ja-JP" },
                    "current_translation": { "text": format!("translation {ordinal}"), "language": "ko-KR" },
                    "logical_dialogue_memberships": [{
                        "group_id": "group-1",
                        "member_ordinal": ordinal,
                        "primary_render_element_id": "element-1",
                    }],
                    "source_polygon": {
                        "points": vec![json!({ "x": 12.0, "y": 34.0 }); 4_000],
                        "bounds": { "x": 12.0, "y": 34.0, "width": 56.0, "height": 78.0 },
                    },
                    "free_dialogue_anchor_assessment": {
                        "source_region_id": format!("region-{ordinal}"),
                        "candidates": (0..32).map(|candidate| json!({
                            "bounds": { "x": candidate, "y": candidate, "width": 80, "height": 40 },
                            "score": 0.91,
                            "accepted": candidate == 0,
                            "pixels": vec![ordinal; 8_000],
                            "rejection_reasons": if candidate == 0 { Vec::<String>::new() } else { vec!["edge_density".to_owned()] },
                        })).collect::<Vec<_>>(),
                    },
                    "original_crop": {
                        "path": format!("/evidence/crop-{ordinal}.png"),
                        "media_type": "image/png",
                        "byte_length": 1234,
                        "blake3": format!("digest-{ordinal}"),
                    },
                })
            })
            .collect::<Vec<_>>();
        let raw = json!({
            "scene_revision": 28,
            "page_id": "page-1",
            "elements": elements,
        });
        let (envelope, compacted) =
            koharu_agent::compact_model_payload(directory.path(), "inspect-source-evidence", raw)
                .unwrap();
        let serialized = serde_json::to_vec(&envelope).unwrap();

        assert!(compacted.original_bytes > koharu_agent::MODEL_RESULT_BUDGET_BYTES);
        assert!(serialized.len() <= koharu_agent::MODEL_RESULT_BUDGET_BYTES);
        assert_eq!(envelope["index"]["scene_revision"], 28);
        assert_eq!(envelope["index"]["elements"][23]["ordinal"], 24);
        assert_eq!(
            envelope["index"]["elements"][23]["source_polygon_bounds"]["width"],
            56.0
        );
        assert_eq!(
            envelope["index"]["elements"][23]["original_crop"]["path"],
            "/evidence/crop-24.png"
        );
        assert!(
            !String::from_utf8(serialized)
                .unwrap()
                .contains("\"points\"")
        );
        let artifact_path = envelope["artifact"]["path"].as_str().unwrap();
        let artifact = std::fs::read(artifact_path).unwrap();
        assert_eq!(
            blake3::hash(&artifact).to_hex().to_string(),
            envelope["artifact"]["blake3"]
        );
    }

    #[test]
    fn simulated_multiphase_route_to_export_stays_below_retained_budget() {
        let directory = tempfile::tempdir().unwrap();
        let phases = [
            "source_analysis",
            "source_evidence",
            "pipeline",
            "page_evidence",
            "deterministic_review",
            "semantic_review",
            "export",
        ];
        let mut transcript = Vec::new();
        let mut observed_compaction = false;
        for turn in 0..42 {
            let phase = phases[turn % phases.len()];
            let raw = json!({
                "scene_revision": turn + 1,
                "page_id": "page-1",
                "status": phase,
                "elements": (1..=16).map(|ordinal| json!({
                    "ordinal": ordinal,
                    "element_id": format!("element-{ordinal}"),
                    "required": true,
                    "current_source": { "text": format!("source-{ordinal}"), "language": "ja-JP" },
                    "current_translation": { "text": format!("target-{ordinal}"), "language": "ko-KR" },
                    "source_polygon": { "points": vec![ordinal; 10_000], "bounds": { "x": 1, "y": 2, "width": 3, "height": 4 } },
                })).collect::<Vec<_>>(),
                "next_action": phases[(turn + 1) % phases.len()],
            });
            let (envelope, _) = koharu_agent::compact_model_payload(
                directory.path(),
                &format!("route-{turn}-{phase}"),
                raw,
            )
            .unwrap();
            transcript.push(json!({
                "type": "function_call_output",
                "call_id": format!("call-{turn}"),
                "output": envelope,
            }));
            transcript.push(json!({
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "x".repeat(18_000) }],
            }));
            observed_compaction |=
                koharu_agent::compact_retained_transcript(directory.path(), &mut transcript)
                    .unwrap()
                    .is_some();
            assert!(
                koharu_agent::retained_transcript_bytes(&transcript).unwrap()
                    <= koharu_agent::RETAINED_TRANSCRIPT_BUDGET_BYTES
            );
        }

        assert!(observed_compaction);
        let retained = serde_json::to_string(&transcript).unwrap();
        assert!(retained.contains("element-16"));
        assert!(retained.contains("\"ordinal\":16"));
        assert!(retained.contains("retained_transcript"));
    }

    #[test]
    fn host_executed_padding_repair_is_not_model_callable() {
        let revision = Revision::new(11);
        let element = EntityId::new();
        let mut record = review(VisualReviewStatus::PendingAgentReview, false, revision);
        record.deterministic_repair_plan = DeterministicRepairPlan {
            schema_version: crate::repair::REPAIR_PLAN_SCHEMA_VERSION,
            unresolved_failure_count: 1,
            blocking_failures: vec![DeterministicRepairFailure {
                element_id: Some(element),
                logical_group_id: None,
                primary_render_element_id: None,
                member_ordinal_ids: Vec::new(),
                code: AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior,
                expected: Value::Null,
                actual: Value::Null,
                allowed_repair_fields: vec![RepairField::Layout],
                next_action: Some(crate::repair::RepairNextAction {
                    tool: "preview_increase_text_safe_padding",
                    operation: "increase_text_safe_padding",
                    element,
                    required_evidence_revision: revision,
                    constraints: Vec::new(),
                }),
                required_evidence_revision: revision,
            }],
            terminal_diagnostic: None,
        };
        let inputs = WorkflowSurfaceInputs {
            source_analysis_completed: true,
            pipeline_completed: true,
            source_evidence_ready: true,
            decorative_sfx_dispositions_complete: true,
            ui_panel_evidence_available: false,
            page_evidence_revision: Some(revision),
            page_evidence_is_current: true,
            current_page_review_accepted: None,
            all_page_reviews_current: false,
            semantic_review_stage: None,
            review: Some(record),
            repair_tool: Some("preview_increase_text_safe_padding"),
            repair_stopped: false,
            exported: false,
        };
        let (phase, exposed) = names(&inputs);
        assert_eq!(phase, WorkflowPhase::ReviewFailed);
        assert_eq!(exposed, ["inspect_project"]);
        for forbidden in [
            "revise_element",
            "revise_page_translation",
            "preview_text_layout",
            "preview_increase_text_safe_padding",
            "preview_compact_translation",
            "export_pages",
        ] {
            assert!(!exposed.contains(&forbidden), "unexpected {forbidden}");
        }

        let tools = exposed
            .iter()
            .map(|name| tool_definition(name).clone())
            .collect::<Vec<_>>();
        let error = ensure_tool_available(phase, &tools, "revise_element").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unavailable during workflow phase review_failed")
        );
    }

    #[test]
    fn fresh_page_clipping_plan_exposes_its_constrained_text_layout_preview() {
        let revision = Revision::new(11);
        let element = EntityId::new();
        let page = EntityId::new();
        let plan = deterministic_repair_plan(
            &[AcceptanceRejection {
                code: AcceptanceRejectionCode::RenderedTextOutsidePage,
                page_id: page,
                element_id: Some(element),
                related_element_id: None,
                expected: json!({ "maximum_px": 0.5 }),
                actual: json!({ "value_px": 9.0 }),
            }],
            revision,
            None,
        );
        let failure = &plan.blocking_failures[0];
        let action = failure.next_action.as_ref().unwrap();
        assert_eq!(action.tool, "preview_text_layout");
        assert_eq!(action.operation, "controlled_text_layout");
        assert!(host_deterministic_repair(failure).unwrap().is_none());

        let mut record = review(VisualReviewStatus::PendingAgentReview, false, revision);
        let repair_tool = action.tool;
        record.deterministic_repair_plan = plan;
        let mut inputs = WorkflowSurfaceInputs {
            source_analysis_completed: true,
            pipeline_completed: true,
            source_evidence_ready: true,
            decorative_sfx_dispositions_complete: true,
            ui_panel_evidence_available: false,
            page_evidence_revision: Some(revision),
            page_evidence_is_current: true,
            current_page_review_accepted: None,
            all_page_reviews_current: false,
            semantic_review_stage: None,
            review: Some(record),
            repair_tool: Some(repair_tool),
            repair_stopped: false,
            exported: false,
        };
        let (phase, exposed) = names(&inputs);
        assert_eq!(phase, WorkflowPhase::DeterministicRepair);
        assert!(exposed.contains(&repair_tool));
        for forbidden in [
            "revise_element",
            "commit_text_layout",
            "preview_increase_text_safe_padding",
            "export_pages",
        ] {
            assert!(!exposed.contains(&forbidden), "unexpected {forbidden}");
        }

        inputs.page_evidence_revision = Some(Revision::new(revision.get() + 1));
        inputs.page_evidence_is_current = false;
        let (phase, exposed) = names(&inputs);
        assert_eq!(phase, WorkflowPhase::DeterministicRepair);
        assert!(!exposed.contains(&repair_tool));
    }

    #[test]
    fn planned_repair_schema_binds_exact_target_and_preview_id() {
        let element = EntityId::new();
        let mut preview = tool_definition("preview_text_layout").clone();
        constrain_repair_tool(
            &mut preview,
            &RepairToolConstraint {
                name: "preview_text_layout",
                operation: Some("controlled_source_raster_verified_free_dialogue_target_layout"),
                element: Some(element),
                logical_group: None,
                page: None,
                revision: Some(Revision::new(3)),
                preview_id: None,
                allowed_fields: vec![RepairField::Layout],
            },
        );
        assert_eq!(
            preview.parameters["properties"]["element"]["const"],
            element.to_string()
        );
        let variants = preview.parameters["$defs"]["TextLayoutOptions"]["anyOf"]
            .as_array()
            .unwrap();
        assert_eq!(variants.len(), 1);
        assert!(
            variants[0]["$ref"]
                .as_str()
                .unwrap()
                .ends_with("ControlledTextLayoutOptions")
        );
        assert!(
            preview
                .description
                .contains("native fallback is not eligible")
        );

        let adjacent_failure = DeterministicRepairFailure {
            element_id: Some(element),
            logical_group_id: None,
            primary_render_element_id: None,
            member_ordinal_ids: Vec::new(),
            code: AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior,
            expected: Value::Null,
            actual: Value::Null,
            allowed_repair_fields: vec![RepairField::Layout],
            next_action: Some(crate::repair::RepairNextAction {
                tool: "preview_text_layout",
                operation: "controlled_source_raster_verified_free_dialogue_target_layout",
                element,
                required_evidence_revision: Revision::new(3),
                constraints: Vec::new(),
            }),
            required_evidence_revision: Revision::new(3),
        };
        assert!(
            validate_text_layout_plan(
                Some(&adjacent_failure),
                &TextLayoutOptions::SourceBoundInterjectionFallback(
                    SourceBoundInterjectionFallbackOptions {
                        source_bound_interjection_fallback:
                            SourceBoundInterjectionFallback::NativeVertical,
                    },
                ),
            )
            .is_err()
        );

        let mut commit = tool_definition("commit_text_layout").clone();
        constrain_repair_tool(
            &mut commit,
            &RepairToolConstraint::commit("commit_text_layout", "preview-1", element),
        );
        assert_eq!(
            commit.parameters["properties"]["preview_id"]["const"],
            "preview-1"
        );
    }

    #[tokio::test]
    async fn successful_text_layout_preview_exposes_only_its_exact_dynamic_commit() {
        let revision = Revision::new(11);
        let element = EntityId::new();
        let page = EntityId::new();
        let plan = deterministic_repair_plan(
            &[AcceptanceRejection {
                code: AcceptanceRejectionCode::RenderedTextOutsidePage,
                page_id: page,
                element_id: Some(element),
                related_element_id: None,
                expected: json!({ "maximum_px": 0.5 }),
                actual: json!({ "value_px": 9.0 }),
            }],
            revision,
            None,
        );
        let action = plan.blocking_failures[0].next_action.as_ref().unwrap();
        let options = TextLayoutOptions::Controlled(ControlledTextLayoutOptions {
            line_break_policy: KoreanLineBreakPolicy::KoreanKeepWordsBalanced,
            max_lines: 3,
            alignment: Some(ControlledTextAlignment::Center),
            font_scale: None,
            safe_padding_increase_px: None,
        });
        assert!(validate_text_layout_plan(Some(&plan.blocking_failures[0]), &options).is_ok());

        let session = Session::memory().await.unwrap();
        let patch = session
            .snapshot()
            .patch(|edit| {
                edit.add_page(PageDraft::new("preview", 100.0, 100.0), At::End)?;
                Ok(())
            })
            .unwrap();
        let preview_id = EntityId::new().to_string();
        let metrics_before = layout_metrics(4, true, 0.0);
        let metrics_candidate = layout_metrics(3, false, 0.0);
        validate_text_layout_candidate(
            &metrics_before,
            &metrics_candidate,
            options.max_lines(),
            &QualityThresholds::default(),
        )
        .unwrap();
        let state = RepairElementState {
            source: None,
            translation: None,
            typography: None,
            layout_kind: TextLayoutKind::Paragraph,
            authored_layout_geometry: None,
        };
        let prepared = PreparedTextLayout {
            preview: TextLayoutPreview {
                preview_id: preview_id.clone(),
                operation: action.operation,
                base_revision: revision,
                element_id: element,
                options,
                metrics_before,
                metrics_candidate,
                deterministic_constraints_satisfied: true,
                mutation_committed: false,
            },
            patch,
            content_id: EntityId::new(),
            evidence: RevisionEvidence::DeterministicRejection {
                revision,
                review_attempt: 1,
            },
            before_state: state.clone(),
            after_state: state,
            reason: "safe production preview".to_owned(),
            changed_fields: vec!["line_break_policy"],
        };
        let mut pending = BTreeMap::new();
        store_pending_text_layout(
            &mut pending,
            "superseded-preview".to_owned(),
            prepared.clone(),
        );
        store_pending_text_layout(&mut pending, preview_id.clone(), prepared);
        assert_eq!(pending.len(), 1);
        assert!(pending.contains_key(&preview_id));

        let mut record = review(VisualReviewStatus::PendingAgentReview, false, revision);
        record.deterministic_repair_plan = plan;
        let commit = pending_text_layout_commit(&record, &pending).unwrap();
        assert_eq!(commit.name, "commit_text_layout");
        assert_eq!(commit.preview_id.as_deref(), Some(preview_id.as_str()));

        let mut inputs = WorkflowSurfaceInputs {
            source_analysis_completed: true,
            pipeline_completed: true,
            source_evidence_ready: true,
            decorative_sfx_dispositions_complete: true,
            ui_panel_evidence_available: false,
            page_evidence_revision: Some(revision),
            page_evidence_is_current: true,
            current_page_review_accepted: None,
            all_page_reviews_current: false,
            semantic_review_stage: None,
            review: Some(record),
            repair_tool: Some(commit.name),
            repair_stopped: false,
            exported: false,
        };
        let (phase, exposed) = names(&inputs);
        assert_eq!(phase, WorkflowPhase::DeterministicRepair);
        assert!(exposed.contains(&"commit_text_layout"));
        for forbidden in [
            "inspect_project",
            "revise_element",
            "revise_page_translation",
            "preview_text_layout",
            "preview_compact_translation",
            "preview_increase_text_safe_padding",
            "commit_text_safe_layout_repair",
            "export_pages",
        ] {
            assert!(!exposed.contains(&forbidden), "unexpected {forbidden}");
        }

        let mut exact_commit = tool_definition("commit_text_layout").clone();
        constrain_repair_tool(&mut exact_commit, &commit);
        assert_eq!(
            exact_commit.parameters["properties"]["preview_id"]["const"],
            preview_id
        );

        inputs.page_evidence_revision = Some(Revision::new(revision.get() + 1));
        inputs.page_evidence_is_current = false;
        assert!(!names(&inputs).1.contains(&"commit_text_layout"));
        pending.clear();
        assert!(pending_text_layout_commit(inputs.review.as_ref().unwrap(), &pending).is_none());
    }

    #[tokio::test]
    async fn validated_compact_translation_preview_exposes_and_dispatches_only_its_exact_commit() {
        let output = tempfile::tempdir().unwrap();
        let review_bundle = tempfile::tempdir().unwrap();
        let host = HarnessHost::create(
            Vec::new(),
            Language::Japanese,
            Language::Korean,
            output.path().to_owned(),
            OutputFormat::Png,
            QualityThresholds::default(),
            review_bundle.path().to_owned(),
            None,
        )
        .await
        .unwrap();
        host.source_analysis_completed
            .store(true, Ordering::Release);
        host.pipeline_completed.store(true, Ordering::Release);

        let snapshot = host.project.session().lock().await.snapshot();
        let revision = snapshot.revision();
        let page = EntityId::new();
        let group = EntityId::new();
        let primary = EntityId::new();
        let secondary = EntityId::new();
        let failure = compact_layout_failure(revision, page, group, primary, secondary);
        let members = failure.member_ordinal_ids.clone();
        let mut review = review(VisualReviewStatus::PendingAgentReview, false, revision);
        review.deterministic_repair_plan = DeterministicRepairPlan {
            schema_version: crate::repair::REPAIR_PLAN_SCHEMA_VERSION,
            unresolved_failure_count: 1,
            blocking_failures: vec![failure],
            terminal_diagnostic: None,
        };
        *host.visual_review.lock() = Some(review.clone());
        host.review_history.lock().push(review);
        *host.page_semantic_evidence.lock() =
            page_semantic_evidence_with_members(revision, page, &members);
        *host.current_semantic_evidence_page.lock() = Some(PageTarget {
            ordinal: PageOrdinal::new(1),
            id: page,
        });
        *host.page_reviews.lock() = PageReviewState::at_revision([page], revision);

        let preview_id = EntityId::new().to_string();
        let before_state = RepairElementState {
            source: None,
            translation: Some(RepairText {
                text: "길어진 대사".to_owned(),
                language: Some("ko-KR".to_owned()),
                origin: koharu_scene::Origin::Generated(Generation::new(
                    ProducerId::new("dev.koharu.test").unwrap(),
                )),
            }),
            typography: None,
            layout_kind: TextLayoutKind::Paragraph,
            authored_layout_geometry: None,
        };
        let mut after_state = before_state.clone();
        after_state.translation.as_mut().unwrap().text = "짧게".to_owned();
        let target_region = EntityId::new();
        let metrics_before = compact_metrics(10.0, false, target_region);
        let metrics_candidate = compact_metrics(13.0, true, target_region);
        validate_compact_translation_candidate(
            &metrics_before,
            &metrics_candidate,
            &QualityThresholds::default(),
        )
        .unwrap();
        let prepared = PreparedCompactTranslation {
            preview: CompactTranslationPreview {
                preview_id: preview_id.clone(),
                operation: "layout_constrained_translation_compaction",
                base_revision: revision,
                logical_group_id: group,
                primary_render_element_id: primary,
                member_ordinal_ids: members.clone(),
                current_visible_grapheme_units: 6,
                candidate_visible_grapheme_units: 2,
                metrics_before,
                metrics_candidate,
                deterministic_constraints_satisfied: true,
                mutation_committed: false,
            },
            patch: snapshot.patch(|_| Ok(())).unwrap(),
            content_id: EntityId::new(),
            evidence: RevisionEvidence::PageTranslationVisualEvidence {
                revision,
                page_id: page,
                source_evidence_dossier_blake3: "source-dossier-digest".to_owned(),
                source_debug_artifact_blake3: "source-debug-digest".to_owned(),
                dossier_blake3: "dossier-digest".to_owned(),
                debug_artifact_blake3: "debug-digest".to_owned(),
            },
            before_state,
            after_state,
            reason: "retain meaning while fitting the reviewed group".to_owned(),
            member_ordinal_ids: members,
        };
        host.pending_compact_translations
            .lock()
            .insert(preview_id.clone(), prepared);

        let (phase, tools) = host.workflow_surface();
        assert_eq!(phase, WorkflowPhase::DeterministicRepair);
        let repair_tools = tools
            .iter()
            .filter(|tool| {
                matches!(
                    tool.name.as_str(),
                    "revise_element"
                        | "revise_page_translation"
                        | "preview_text_layout"
                        | "commit_text_layout"
                        | "preview_compact_translation"
                        | "commit_compact_translation"
                        | "preview_increase_text_safe_padding"
                        | "commit_text_safe_layout_repair"
                        | "export_pages"
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(repair_tools.len(), 1);
        assert_eq!(repair_tools[0].name, "commit_compact_translation");
        assert_eq!(
            repair_tools[0].parameters["properties"]["preview_id"]["const"],
            preview_id
        );

        let error = host
            .invoke(
                ToolCall {
                    call_id: "dispatch-compact-translation-commit".to_owned(),
                    name: "commit_compact_translation".to_owned(),
                    arguments: json!({ "preview_id": preview_id }).to_string(),
                },
                &Control::default(),
            )
            .await
            .unwrap_err();
        assert!(
            !error
                .to_string()
                .contains("unknown Koharu harness tool commit_compact_translation"),
            "commit dispatch fell through to the unknown-tool branch: {error:#}"
        );
    }

    #[tokio::test]
    async fn nine_page_source_evidence_tools_address_the_seventh_page_by_ordinal() {
        let fixture = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        let review_bundle = tempfile::tempdir().unwrap();
        let mut encoded = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            32,
            32,
            image::Rgba([255, 255, 255, 255]),
        ))
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
        let page_bytes = encoded.into_inner();
        let inputs = (1..=9)
            .map(|ordinal| {
                let input = fixture.path().join(format!("page-{ordinal:02}.png"));
                std::fs::write(&input, &page_bytes).unwrap();
                input
            })
            .collect::<Vec<_>>();
        let host = HarnessHost::create(
            inputs,
            Language::Japanese,
            Language::Korean,
            output.path().to_owned(),
            OutputFormat::Png,
            QualityThresholds::default(),
            review_bundle.path().to_owned(),
            None,
        )
        .await
        .unwrap();
        host.source_analysis_completed
            .store(true, Ordering::Release);
        let page_ids = host
            .project
            .session()
            .lock()
            .await
            .snapshot()
            .pages()
            .map(|page| page.id())
            .collect::<Vec<_>>();

        host.invoke(
            ToolCall {
                call_id: "inspect-page-seven-source".to_owned(),
                name: "inspect_source_evidence".to_owned(),
                arguments: json!({ "page_ordinal": 7 }).to_string(),
            },
            &Control::default(),
        )
        .await
        .unwrap();
        host.invoke(
            ToolCall {
                call_id: "debug-page-seven-source".to_owned(),
                name: "view_page_source_debug".to_owned(),
                arguments: json!({ "page_ordinal": 7 }).to_string(),
            },
            &Control::default(),
        )
        .await
        .unwrap();

        let evidence = host.page_semantic_evidence.lock();
        let addressed = evidence.page(page_ids[6]).unwrap();
        assert!(addressed.source_dossier.is_some());
        assert!(addressed.source_debug_artifact.is_some());
        for neighbor in [page_ids[5], page_ids[7]] {
            assert!(
                evidence.page(neighbor).is_none(),
                "neighboring page {neighbor} unexpectedly received source evidence"
            );
        }
    }

    #[tokio::test]
    async fn source_phase_keeps_the_exact_pending_candidate_across_pages_and_partial_disposition() {
        let fixture = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        let review_bundle = tempfile::tempdir().unwrap();
        let mut encoded = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            120,
            120,
            image::Rgba([255, 255, 255, 255]),
        ))
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
        let page_bytes = encoded.into_inner();
        let inputs = (1..=2)
            .map(|ordinal| {
                let input = fixture.path().join(format!("page-{ordinal:02}.png"));
                std::fs::write(&input, &page_bytes).unwrap();
                input
            })
            .collect::<Vec<_>>();
        let host = HarnessHost::create(
            inputs,
            Language::Japanese,
            Language::Korean,
            output.path().to_owned(),
            OutputFormat::Png,
            QualityThresholds::default(),
            review_bundle.path().to_owned(),
            None,
        )
        .await
        .unwrap();

        let generation = Generation::new(ProducerId::new("dev.koharu.test.detector").unwrap());
        let mut session = host.project.session().lock().await;
        let snapshot = session.snapshot();
        let pages = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
        let mut edit = snapshot.edit_as(generation.clone());
        let mut candidates = Vec::new();
        for (page, x, y, source) in [
            (pages[0], 80.0, 10.0, "first"),
            (pages[0], 60.0, 30.0, "remaining"),
            (pages[1], 70.0, 20.0, "later"),
        ] {
            let source_region = edit
                .add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(x, y, 20.0, 24.0),
                    Some("detected free text".to_owned()),
                )
                .unwrap();
            edit.set(
                source_region,
                &DetectionAnalysis {
                    origin: koharu_scene::Origin::Generated(generation.clone()),
                    labels: vec![DetectionLabel {
                        kind: TextRegion::kind(),
                        confidence: 0.98,
                    }],
                },
            )
            .unwrap();
            edit.set(
                source_region,
                &OcrAnalysis {
                    origin: koharu_scene::Origin::Generated(generation.clone()),
                    direction: TextDirection::Vertical,
                    confidence: Some(0.98),
                    line_boundaries: Vec::new(),
                },
            )
            .unwrap();
            let content = edit.add_text_content(page, At::End).unwrap();
            let element = edit
                .add_text_layer(
                    page,
                    At::End,
                    content,
                    &TextLayout {
                        origin: koharu_scene::Origin::User,
                        kind: TextLayoutKind::Paragraph,
                    },
                )
                .unwrap();
            edit.relate::<koharu_scene::RecognizedFrom>(content, source_region)
                .unwrap();
            edit.relate::<FitsTo>(element, source_region).unwrap();
            edit.set(
                content,
                &SourceText {
                    text: Authored::user(source.to_owned()),
                    language: Some(LanguageTag::new("ja-JP").unwrap()),
                },
            )
            .unwrap();
            edit.set(
                content,
                &TextRole {
                    origin: koharu_scene::Origin::User,
                    role: FREE_TEXT_ROLE.to_owned(),
                },
            )
            .unwrap();
            edit.set(
                element,
                &Typography {
                    origin: koharu_scene::Origin::User,
                    preferred_font: None,
                    font_weight: None,
                    font_style: None,
                    size: Some(18.0),
                    auto_fit: true,
                    color: Some([0, 0, 0, 255]),
                    stroke_color: Some([255, 255, 255, 255]),
                    stroke_width: Some(1.0),
                    alignment: None,
                    writing_mode: Some(WritingMode::Vertical),
                    extensions: Default::default(),
                },
            )
            .unwrap();
            edit.set(element, &Geometry::rectangle(x, y, 20.0, 24.0))
                .unwrap();
            candidates.push(element);
        }
        session.commit(edit.finish().unwrap()).await.unwrap();
        drop(session);

        host.source_analysis_completed
            .store(true, Ordering::Release);
        *host.pending_decorative_sfx_dispositions.lock() =
            BTreeSet::from_iter(candidates.iter().copied());

        async fn evidence_for(host: &HarnessHost, page_ordinal: usize) -> (Invocation, Invocation) {
            let source = host
                .invoke(
                    ToolCall {
                        call_id: format!("inspect-page-{page_ordinal}"),
                        name: "inspect_source_evidence".to_owned(),
                        arguments: json!({ "page_ordinal": page_ordinal }).to_string(),
                    },
                    &Control::default(),
                )
                .await
                .unwrap();
            let debug = host
                .invoke(
                    ToolCall {
                        call_id: format!("debug-page-{page_ordinal}"),
                        name: "view_page_source_debug".to_owned(),
                        arguments: json!({ "page_ordinal": page_ordinal }).to_string(),
                    },
                    &Control::default(),
                )
                .await
                .unwrap();
            (source, debug)
        }

        async fn retain(
            host: &HarnessHost,
            page_ordinal: usize,
            element: EntityId,
            source: &Invocation,
            debug: &Invocation,
        ) -> Invocation {
            let evidence = source.value["source_evidence"]["elements"]
                .as_array()
                .unwrap()
                .iter()
                .find(|candidate| candidate["element_id"] == element.to_string())
                .unwrap();
            host.invoke(
                ToolCall {
                    call_id: format!("retain-{element}"),
                    name: "classify_decorative_sfx".to_owned(),
                    arguments: json!({
                        "page_ordinal": page_ordinal,
                        "scene_revision": source.value["source_evidence"]["scene_revision"],
                        "source_evidence_dossier_blake3": source.value["source_evidence_dossier_blake3"],
                        "source_debug_artifact_blake3": debug.value["blake3"],
                        "decisions": [{
                            "element": element,
                            "original_ordinal": evidence["ordinal"],
                            "disposition": "retain_required",
                            "source_crop_blake3": evidence["original_crop"]["blake3"],
                            "source_debug_label": evidence["source_debug_label"],
                            "confidence": 0.98,
                            "evidence": {
                                "decorative_visual_form": "ordinary compact speech lettering without decorative effects",
                                "sound_effect_page_function": "the crop functions as a character utterance rather than a sound effect",
                                "legibility_and_translation_value": "the utterance is legible and carries ordinary dialogue meaning",
                                "exclusion_of_dialogue_caption_ui_and_general_free_text": "ordinary free text is visibly present, excluding decorative SFX"
                            },
                            "rationale": "Fresh source crop and debug evidence establish required ordinary text."
                        }]
                    })
                    .to_string(),
                },
                &Control::default(),
            )
            .await
            .unwrap()
        }

        let (page_one_source, page_one_debug) = evidence_for(&host, 1).await;
        assert_eq!(
            page_one_source.value["source_evidence"]["pending_decorative_sfx_dispositions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|candidate| candidate["element_id"].as_str().unwrap().to_owned())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([candidates[0].to_string(), candidates[1].to_string(),])
        );
        retain(&host, 1, candidates[0], &page_one_source, &page_one_debug).await;

        let obligation = format!(
            "page ordinal 1, page ID {}, element {}, original ordinal 2, required evidence stage classify_decorative_sfx",
            pages[0], candidates[1]
        );
        let tools = host.tools();
        for name in ["inspect_source_evidence", "view_page_source_debug"] {
            let tool = tools.iter().find(|tool| tool.name == name).unwrap();
            assert_eq!(tool.parameters["properties"]["page_ordinal"]["const"], 1);
            assert!(
                tool.description.contains(&obligation),
                "{name}: {}",
                tool.description
            );
        }
        assert!(!tools.iter().any(|tool| tool.name == "run_pipeline"));

        let revision_before_wrong_page = host.project.session().lock().await.snapshot().revision();
        let evidence_page_before_wrong_page = *host.current_semantic_evidence_page.lock();
        let pending_before_wrong_page = host.pending_decorative_sfx_dispositions.lock().clone();
        for (name, arguments) in [
            (
                "inspect_source_evidence",
                json!({ "page_ordinal": 2 }).to_string(),
            ),
            (
                "classify_decorative_sfx",
                json!({
                    "page_ordinal": 2,
                    "scene_revision": revision_before_wrong_page,
                    "source_evidence_dossier_blake3": page_one_source.value["source_evidence_dossier_blake3"],
                    "source_debug_artifact_blake3": page_one_debug.value["blake3"],
                    "decisions": []
                })
                .to_string(),
            ),
        ] {
            let error = host
                .invoke(
                    ToolCall {
                        call_id: format!("wrong-page-{name}"),
                        name: name.to_owned(),
                        arguments,
                    },
                    &Control::default(),
                )
                .await
                .unwrap_err();
            assert!(error.to_string().contains(&obligation), "{error:#}");
        }
        assert_eq!(
            host.project.session().lock().await.snapshot().revision(),
            revision_before_wrong_page
        );
        let evidence_page_after_wrong_page = *host.current_semantic_evidence_page.lock();
        assert_eq!(
            evidence_page_after_wrong_page.map(|page| (page.ordinal.get(), page.id)),
            evidence_page_before_wrong_page.map(|page| (page.ordinal.get(), page.id))
        );
        assert_eq!(
            *host.pending_decorative_sfx_dispositions.lock(),
            pending_before_wrong_page
        );
        assert!(host.page_semantic_evidence.lock().page(pages[1]).is_none());

        match host.completion().await.unwrap() {
            HostCompletion::Continue {
                phase,
                exposed_tools,
                reason,
                ..
            } => {
                assert_eq!(phase, "source_evidence");
                assert!(exposed_tools.contains(&"inspect_source_evidence".to_owned()));
                assert!(reason.contains(&obligation), "{reason}");
            }
            HostCompletion::Completed => panic!("pending source work must continue"),
        }

        let (page_one_source, page_one_debug) = evidence_for(&host, 1).await;
        let classify = host
            .tools()
            .into_iter()
            .find(|tool| tool.name == "classify_decorative_sfx")
            .unwrap();
        assert_eq!(
            classify.parameters["properties"]["page_ordinal"]["const"],
            1
        );
        assert_eq!(
            classify.parameters["$defs"]["DecorativeSfxClassification"]["properties"]["element"]["enum"],
            json!([candidates[1].to_string()])
        );
        assert!(classify.description.contains(&obligation));
        retain(&host, 1, candidates[1], &page_one_source, &page_one_debug).await;

        let (page_two_source, page_two_debug) = evidence_for(&host, 2).await;
        retain(&host, 2, candidates[2], &page_two_source, &page_two_debug).await;

        assert!(host.pending_decorative_sfx_dispositions.lock().is_empty());
        assert!(host.tools().iter().any(|tool| tool.name == "run_pipeline"));
    }

    #[tokio::test]
    async fn run_pipeline_remains_exposed_when_current_source_evidence_page_lacks_debug() {
        let fixture = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        let review_bundle = tempfile::tempdir().unwrap();
        let mut encoded = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            32,
            32,
            image::Rgba([255, 255, 255, 255]),
        ))
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
        let page_bytes = encoded.into_inner();
        let inputs = (1..=2)
            .map(|ordinal| {
                let input = fixture.path().join(format!("page-{ordinal:02}.png"));
                std::fs::write(&input, &page_bytes).unwrap();
                input
            })
            .collect::<Vec<_>>();
        let host = HarnessHost::create(
            inputs,
            Language::Japanese,
            Language::Korean,
            output.path().to_owned(),
            OutputFormat::Png,
            QualityThresholds::default(),
            review_bundle.path().to_owned(),
            None,
        )
        .await
        .unwrap();
        let inspection = host.inspect_project().await.unwrap();
        let pending_screening = decorative_sfx_disposition_candidates(&inspection);
        assert!(pending_screening.is_empty());
        *host.pending_decorative_sfx_dispositions.lock() = pending_screening;
        host.source_analysis_completed
            .store(true, Ordering::Release);
        let page_ids = host
            .project
            .session()
            .lock()
            .await
            .snapshot()
            .pages()
            .map(|page| page.id())
            .collect::<Vec<_>>();

        for (call_id, name) in [
            ("inspect-page-one-source", "inspect_source_evidence"),
            ("debug-page-one-source", "view_page_source_debug"),
        ] {
            host.invoke(
                ToolCall {
                    call_id: call_id.to_owned(),
                    name: name.to_owned(),
                    arguments: json!({ "page_ordinal": 1 }).to_string(),
                },
                &Control::default(),
            )
            .await
            .unwrap();
        }
        assert!(host.tools().iter().any(|tool| tool.name == "run_pipeline"));

        host.invoke(
            ToolCall {
                call_id: "inspect-page-two-source".to_owned(),
                name: "inspect_source_evidence".to_owned(),
                arguments: json!({ "page_ordinal": 2 }).to_string(),
            },
            &Control::default(),
        )
        .await
        .unwrap();

        let current_page = host
            .current_semantic_evidence_page
            .lock()
            .expect("source evidence inspection should select its page");
        assert_eq!(current_page.ordinal.get(), 2);
        assert_eq!(current_page.id, page_ids[1]);
        let evidence = host.page_semantic_evidence.lock();
        let current_page_evidence = evidence.page(page_ids[1]).unwrap();
        assert!(current_page_evidence.source_dossier.is_some());
        assert!(current_page_evidence.source_debug_artifact.is_none());
        drop(evidence);

        let tools = host.tools();
        assert!(
            !tools
                .iter()
                .any(|tool| tool.name == "classify_decorative_sfx")
        );
        assert!(
            tools.iter().any(|tool| tool.name == "run_pipeline"),
            "run_pipeline must remain exposed after all screening dispositions are complete"
        );
    }

    #[tokio::test]
    async fn retain_required_clears_screening_and_translated_revision_refreshes_source_ocr() {
        let fixture = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        let review_bundle = tempfile::tempdir().unwrap();
        let input = fixture.path().join("page.png");
        let mut bytes = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            100,
            100,
            image::Rgba([255, 255, 255, 255]),
        ))
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
        std::fs::write(&input, bytes.into_inner()).unwrap();
        let host = HarnessHost::create(
            vec![input],
            Language::Japanese,
            Language::Korean,
            output.path().to_owned(),
            OutputFormat::Png,
            QualityThresholds::default(),
            review_bundle.path().to_owned(),
            None,
        )
        .await
        .unwrap();

        let generation = Generation::new(ProducerId::new("dev.koharu.test.detector").unwrap());
        let mut ids = Vec::new();
        let mut session = host.project.session().lock().await;
        let snapshot = session.snapshot();
        let page = snapshot.pages().next().unwrap().id();
        let patch = {
            let mut edit = snapshot.edit_as(generation.clone());
            for (x, source) in [(10.0, "え～"), (50.0, "スンストーツイ")] {
                let source_region = edit
                    .add_analysis_region::<TextRegion>(
                        page,
                        At::End,
                        &Geometry::rectangle(x, 10.0, 20.0, 30.0),
                        Some("detected free text".to_owned()),
                    )
                    .unwrap();
                edit.set(
                    source_region,
                    &DetectionAnalysis {
                        origin: koharu_scene::Origin::Generated(generation.clone()),
                        labels: vec![DetectionLabel {
                            kind: TextRegion::kind(),
                            confidence: 0.98,
                        }],
                    },
                )
                .unwrap();
                edit.set(
                    source_region,
                    &OcrAnalysis {
                        origin: koharu_scene::Origin::Generated(generation.clone()),
                        direction: TextDirection::Vertical,
                        confidence: Some(0.98),
                        line_boundaries: Vec::new(),
                    },
                )
                .unwrap();
                let content = edit.add_text_content(page, At::End).unwrap();
                let element = edit
                    .add_text_layer(
                        page,
                        At::End,
                        content,
                        &TextLayout {
                            origin: koharu_scene::Origin::User,
                            kind: TextLayoutKind::Paragraph,
                        },
                    )
                    .unwrap();
                edit.relate::<koharu_scene::RecognizedFrom>(content, source_region)
                    .unwrap();
                edit.relate::<FitsTo>(element, source_region).unwrap();
                edit.set(
                    content,
                    &SourceText {
                        text: Authored::user(source.to_owned()),
                        language: Some(LanguageTag::new("ja-JP").unwrap()),
                    },
                )
                .unwrap();
                edit.set(
                    content,
                    &TextRole {
                        origin: koharu_scene::Origin::User,
                        role: FREE_TEXT_ROLE.to_owned(),
                    },
                )
                .unwrap();
                edit.set(
                    element,
                    &Typography {
                        origin: koharu_scene::Origin::User,
                        preferred_font: None,
                        font_weight: None,
                        font_style: None,
                        size: Some(18.0),
                        auto_fit: true,
                        color: Some([0, 0, 0, 255]),
                        stroke_color: Some([255, 255, 255, 255]),
                        stroke_width: Some(1.0),
                        alignment: None,
                        writing_mode: Some(WritingMode::Vertical),
                        extensions: Default::default(),
                    },
                )
                .unwrap();
                edit.set(element, &Geometry::rectangle(x, 10.0, 20.0, 30.0))
                    .unwrap();
                ids.push((source_region, content, element));
            }
            edit.finish().unwrap()
        };
        session.commit(patch).await.unwrap();
        drop(session);
        let (_retained_source_region, _retained_content_id, retained_element_id) = ids[0];
        let (_source_region, content_id, element_id) = ids[1];

        host.source_analysis_completed
            .store(true, Ordering::Release);
        let before = host.inspect_project().await.unwrap();
        let candidates = decorative_sfx_disposition_candidates(&before);
        assert_eq!(
            candidates,
            BTreeSet::from([retained_element_id, element_id])
        );
        *host.pending_decorative_sfx_dispositions.lock() = candidates;
        let before_element = before.project.pages[0]
            .text_elements
            .iter()
            .find(|element| element.id == retained_element_id)
            .unwrap();
        let before_visibility = (
            before_element.visibility.local_visible,
            before_element.visibility.local_opacity,
            before_element.visibility.effective_visible,
            before_element.visibility.effective_opacity,
        );

        let source_evidence = host
            .invoke(
                ToolCall {
                    call_id: "inspect-translated-sfx".to_owned(),
                    name: "inspect_source_evidence".to_owned(),
                    arguments: json!({ "page_ordinal": 1 }).to_string(),
                },
                &Control::default(),
            )
            .await
            .unwrap();
        let source_debug = host
            .invoke(
                ToolCall {
                    call_id: "debug-translated-sfx".to_owned(),
                    name: "view_page_source_debug".to_owned(),
                    arguments: json!({ "page_ordinal": 1 }).to_string(),
                },
                &Control::default(),
            )
            .await
            .unwrap();
        let evidence_elements = source_evidence.value["source_evidence"]["elements"]
            .as_array()
            .unwrap();
        let retained_evidence = evidence_elements
            .iter()
            .find(|candidate| candidate["element_id"] == retained_element_id.to_string())
            .unwrap();
        let translated_evidence = evidence_elements
            .iter()
            .find(|candidate| candidate["element_id"] == element_id.to_string())
            .unwrap();
        let screening_tools = host.tools();
        assert!(
            !screening_tools
                .iter()
                .any(|tool| tool.name == "run_pipeline")
        );
        assert!(screening_tools.iter().any(|tool| {
            tool.name == "classify_decorative_sfx"
                && tool
                    .description
                    .contains("Current pending screening candidates: 2.")
        }));
        let decision = host
            .invoke(
                ToolCall {
                    call_id: "translate-decorative-sfx".to_owned(),
                    name: "classify_decorative_sfx".to_owned(),
                    arguments: json!({
                        "page_ordinal": 1,
                        "scene_revision": before.project.revision,
                        "source_evidence_dossier_blake3": source_evidence.value["source_evidence_dossier_blake3"],
                        "source_debug_artifact_blake3": source_debug.value["blake3"],
                        "decisions": [
                            {
                                "element": retained_element_id,
                                "original_ordinal": retained_evidence["ordinal"],
                                "disposition": "retain_required",
                                "source_crop_blake3": retained_evidence["original_crop"]["blake3"],
                                "source_debug_label": retained_evidence["source_debug_label"],
                                "confidence": 0.98,
                                "evidence": {
                                    "decorative_visual_form": "ordinary compact speech lettering without decorative effects",
                                    "sound_effect_page_function": "the crop functions as a character reaction utterance, not a sound effect",
                                    "legibility_and_translation_value": "the reaction is legible and carries ordinary dialogue meaning",
                                    "exclusion_of_dialogue_caption_ui_and_general_free_text": "dialogue is visibly present, so the decorative-SFX exclusion criterion is not met"
                                },
                                "rationale": "Fresh source crop and debug evidence establish required reaction dialogue rather than decorative SFX."
                            },
                            {
                                "element": element_id,
                                "original_ordinal": translated_evidence["ordinal"],
                                "disposition": "translate",
                                "source_crop_blake3": translated_evidence["original_crop"]["blake3"],
                                "source_debug_label": translated_evidence["source_debug_label"],
                                "confidence": 0.98,
                                "evidence": {
                                    "decorative_visual_form": "large stylized lettering follows the depicted impact",
                                    "sound_effect_page_function": "the lettering communicates an impact sound",
                                    "legibility_and_translation_value": "the sound effect is legible and should be translated",
                                    "exclusion_of_dialogue_caption_ui_and_general_free_text": "the detached impact lettering is not dialogue, caption, UI, or general free text"
                                },
                                "rationale": "Fresh source crop and debug evidence establish a translated decorative sound effect."
                            }
                        ]
                    })
                    .to_string(),
                },
                &Control::default(),
            )
            .await
            .unwrap();

        assert_eq!(
            decision.value["decisions"][0]["disposition"],
            "retain_required"
        );
        assert_eq!(
            decision.value["decisions"][0]["schema_version"],
            DECORATIVE_SFX_DECISION_SCHEMA_VERSION
        );
        assert_eq!(
            decision.value["decisions"][0]["review_state"],
            "retained_required_non_decorative_sfx"
        );
        assert_eq!(decision.value["decisions"][1]["disposition"], "translate");
        assert_eq!(
            decision.value["decisions"][1]["review_state"],
            "translated_decorative_sfx"
        );
        assert_eq!(decision.value["pending_screening_candidates"], 0);
        assert_eq!(
            decision.value["next_action"],
            "no screening candidates remain; refresh source evidence for the new scene revision, then run_pipeline"
        );
        assert!(host.pending_decorative_sfx_dispositions.lock().is_empty());
        assert!(
            host.tools()
                .iter()
                .any(|tool| tool.name == "inspect_source_evidence")
        );
        let after = host.inspect_project().await.unwrap();
        let retained_element = after.project.pages[0]
            .text_elements
            .iter()
            .find(|element| element.id == retained_element_id)
            .unwrap();
        assert!(retained_element.required);
        assert_eq!(retained_element.text_role.as_deref(), Some(FREE_TEXT_ROLE));
        assert_eq!(
            (
                retained_element.visibility.local_visible,
                retained_element.visibility.local_opacity,
                retained_element.visibility.effective_visible,
                retained_element.visibility.effective_opacity,
            ),
            before_visibility
        );
        let translated_element = after.project.pages[0]
            .text_elements
            .iter()
            .find(|element| element.id == element_id)
            .unwrap();
        assert_eq!(
            translated_element.text_role.as_deref(),
            Some(DECORATIVE_SFX_ROLE)
        );
        let acceptance = evaluate(&after);
        assert!(!acceptance.rejection_reasons.iter().any(|reason| {
            reason.element_id == Some(retained_element_id)
                && reason.code == AcceptanceRejectionCode::InvalidDecorativeSfxDecision
        }));
        assert!(acceptance.rejection_reasons.iter().any(|reason| {
            reason.element_id == Some(retained_element_id)
                && reason.code == AcceptanceRejectionCode::TranslationMissing
        }));

        let mut session = host.project.session().lock().await;
        let snapshot = session.snapshot();
        let patch = snapshot
            .patch(|edit| {
                edit.set(
                    content_id,
                    &Translation {
                        text: Authored::user("에~".to_owned()),
                        language: Some(LanguageTag::new("ko-KR")?),
                    },
                )?;
                Ok(())
            })
            .unwrap();
        let pipeline_revision = session.commit(patch).await.unwrap().snapshot.revision();
        drop(session);
        host.record_page_mutation(page, pipeline_revision);
        host.pipeline_completed.store(true, Ordering::Release);

        async fn revision_evidence(host: &HarnessHost) -> Value {
            let source = host
                .inspect_source_evidence(&ToolCall {
                    call_id: "inspect-revision-source".to_owned(),
                    name: "inspect_source_evidence".to_owned(),
                    arguments: json!({ "page_ordinal": 1 }).to_string(),
                })
                .await
                .unwrap();
            let source_debug = host
                .view_page_source_debug(&ToolCall {
                    call_id: "debug-revision-source".to_owned(),
                    name: "view_page_source_debug".to_owned(),
                    arguments: json!({ "page_ordinal": 1 }).to_string(),
                })
                .await
                .unwrap();
            let translation = host
                .review_page_translation(&ToolCall {
                    call_id: "review-revision-translation".to_owned(),
                    name: "review_page_translation".to_owned(),
                    arguments: json!({ "page_ordinal": 1 }).to_string(),
                })
                .await
                .unwrap();
            let debug = host
                .view_page_debug(&ToolCall {
                    call_id: "debug-revision-translation".to_owned(),
                    name: "view_page_debug".to_owned(),
                    arguments: json!({ "page_ordinal": 1 }).to_string(),
                })
                .await
                .unwrap();
            json!({
                "scene_revision": source.value["source_evidence"]["scene_revision"],
                "source_evidence_dossier_blake3": source.value["source_evidence_dossier_blake3"],
                "source_debug_artifact_blake3": source_debug.value["blake3"],
                "dossier_blake3": translation.value["dossier_blake3"],
                "debug_artifact_blake3": debug.value["blake3"],
            })
        }

        let decision_before_translation_only = host
            .decorative_sfx_decisions
            .lock()
            .get(&element_id)
            .unwrap()
            .clone();
        let _translation_only = host
            .revise_page_translation(&ToolCall {
                call_id: "translation-only-sfx-revision".to_owned(),
                name: "revise_page_translation".to_owned(),
                arguments: serde_json::to_string(&json!({
                    "page_ordinal": 1,
                    "evidence": revision_evidence(&host).await,
                    "edits": [{
                        "element": element_id,
                        "translation": { "text": "쾅!", "language": "ko-KR" }
                    }],
                    "page_rationale": "Correct the translated sound effect without changing its source semantics."
                }))
                .unwrap(),
            })
            .await
            .unwrap();
        let decision_after_translation_only = host
            .decorative_sfx_decisions
            .lock()
            .get(&element_id)
            .unwrap()
            .clone();
        assert_eq!(
            decision_after_translation_only.source_ocr.text,
            decision_before_translation_only.source_ocr.text
        );
        assert_eq!(
            decision_after_translation_only.decision_revision,
            decision_before_translation_only.decision_revision
        );

        let semantic_correction = host
            .revise_page_translation(&ToolCall {
                call_id: "source-and-translation-sfx-revision".to_owned(),
                name: "revise_page_translation".to_owned(),
                arguments: serde_json::to_string(&json!({
                    "page_ordinal": 1,
                    "evidence": revision_evidence(&host).await,
                    "edits": [{
                        "element": element_id,
                        "source": { "text": "ドン", "language": "ja-JP" },
                        "translation": { "text": "쾅", "language": "ko-KR" }
                    }],
                    "page_rationale": "The authoritative source crop shows ドン, not the original OCR text."
                }))
                .unwrap(),
            })
            .await
            .unwrap();
        let committed_revision = Revision::new(
            semantic_correction.value["revision_after"]
                .as_u64()
                .unwrap(),
        );
        let refreshed_decision = host
            .decorative_sfx_decisions
            .lock()
            .get(&element_id)
            .unwrap()
            .clone();
        assert_eq!(refreshed_decision.source_ocr.text, "ドン");
        assert_eq!(
            refreshed_decision.source_ocr.language.as_deref(),
            Some("ja-JP")
        );
        assert_eq!(refreshed_decision.decision_revision, committed_revision);
        let after = host.inspect_project().await.unwrap();
        let acceptance = evaluate(&after);
        assert!(!acceptance.rejection_reasons.iter().any(|reason| {
            reason.element_id == Some(element_id)
                && reason.code == AcceptanceRejectionCode::InvalidDecorativeSfxDecision
        }));

        let retained_semantic_correction = host
            .revise_page_translation(&ToolCall {
                call_id: "source-and-translation-retained-revision".to_owned(),
                name: "revise_page_translation".to_owned(),
                arguments: serde_json::to_string(&json!({
                    "page_ordinal": 1,
                    "evidence": revision_evidence(&host).await,
                    "edits": [{
                        "element": retained_element_id,
                        "source": { "text": "えー", "language": "ja-JP" },
                        "translation": { "text": "어~", "language": "ko-KR" }
                    }],
                    "page_rationale": "The authoritative source crop corrects the retained required dialogue OCR and its translation."
                }))
                .unwrap(),
            })
            .await
            .unwrap();
        let retained_committed_revision = Revision::new(
            retained_semantic_correction.value["revision_after"]
                .as_u64()
                .unwrap(),
        );
        let refreshed_retained_decision = host
            .decorative_sfx_decisions
            .lock()
            .get(&retained_element_id)
            .unwrap()
            .clone();
        assert_eq!(refreshed_retained_decision.source_ocr.text, "えー");
        assert_eq!(
            refreshed_retained_decision.source_ocr.language.as_deref(),
            Some("ja-JP")
        );
        assert_eq!(
            refreshed_retained_decision.decision_revision,
            retained_committed_revision
        );
        let after = host.inspect_project().await.unwrap();
        let acceptance = evaluate(&after);
        assert!(!acceptance.rejection_reasons.iter().any(|reason| {
            reason.element_id == Some(retained_element_id)
                && reason.code == AcceptanceRejectionCode::InvalidDecorativeSfxDecision
        }));
    }

    #[test]
    fn serialized_initial_prompt_and_tool_surface_stay_below_context_budget() {
        let prompt = crate::harness_prompt(
            "Translate every imported page.",
            Language::Japanese,
            Language::Korean,
            OutputFormat::Png,
        );
        let inputs = surface_inputs();
        let (_, names) = names(&inputs);
        let tools = names
            .iter()
            .map(|name| tool_definition(name).clone())
            .collect::<Vec<_>>();
        let serialized = serde_json::to_vec(&json!({
            "prompt": prompt,
            "tools": tools,
        }))
        .unwrap();
        // Three UTF-8 bytes per token is deliberately conservative for mixed prose/schema JSON.
        let estimated_tokens = serialized.len().div_ceil(3);
        let safe_limit = crate::INITIAL_CONTEXT_BUDGET_ESTIMATED_TOKENS * 3 / 4;
        assert!(
            estimated_tokens <= safe_limit,
            "initial request estimate {estimated_tokens} exceeds safe limit {safe_limit} (hard budget {})",
            crate::INITIAL_CONTEXT_BUDGET_ESTIMATED_TOKENS
        );
        assert!(tools.iter().all(|tool| tool.description.len() < 400));
    }

    fn sfx_classification_page(role: &str, source: &str) -> PageInspection {
        let source_region_id = EntityId::new();
        let geometry = ElementGeometry {
            points: vec![
                ElementPoint { x: 10.0, y: 10.0 },
                ElementPoint { x: 30.0, y: 10.0 },
                ElementPoint { x: 30.0, y: 40.0 },
                ElementPoint { x: 10.0, y: 40.0 },
            ],
            bounds: ElementBounds {
                x: 10.0,
                y: 10.0,
                width: 20.0,
                height: 30.0,
            },
        };
        PageInspection {
            id: EntityId::new(),
            label: "page.png".to_owned(),
            width: 100.0,
            height: 100.0,
            text_elements: vec![TextElementInspection {
                id: EntityId::new(),
                content_id: EntityId::new(),
                source_region_id: Some(source_region_id),
                source_region_kind: Some("dev.koharu.region.text".to_owned()),
                detected: true,
                required: true,
                text_role: Some(role.to_owned()),
                decorative_sfx: None,
                logical_dialogue_memberships: Vec::new(),
                source: Some(SemanticText {
                    text: source.to_owned(),
                    language: Some("ja".to_owned()),
                }),
                translation: None,
                source_writing_mode: Some(WritingMode::Vertical),
                visibility: ElementVisibility {
                    local_visible: true,
                    local_opacity: 1.0,
                    effective_visible: true,
                    effective_opacity: 1.0,
                },
                source_geometry: Some(geometry.clone()),
                text_safe_region: Some(TextSafeRegion {
                    id: source_region_id,
                    kind: "dev.koharu.region.text".to_owned(),
                    geometry,
                    association: None,
                }),
                verified_ui_panel_anchor: None,
                verified_free_dialogue_anchor: None,
                free_dialogue_anchor_assessment: None,
                typography: Some(Typography {
                    origin: koharu_scene::Origin::User,
                    preferred_font: None,
                    font_weight: None,
                    font_style: None,
                    size: Some(18.0),
                    auto_fit: true,
                    color: Some([0, 0, 0, 255]),
                    stroke_color: Some([255, 255, 255, 255]),
                    stroke_width: Some(1.0),
                    alignment: None,
                    writing_mode: Some(WritingMode::Vertical),
                    extensions: Default::default(),
                }),
                layout_kind: TextLayoutKind::Paragraph,
                authored_layout_geometry: None,
                final_scene: FinalSceneElement::default(),
            }],
            detected_panel_candidates: Vec::new(),
            logical_dialogue_groups: Vec::new(),
            render_error: None,
        }
    }

    fn sfx_classification_fixture(
        page: &PageInspection,
    ) -> (PageSemanticEvidenceState, ClassifyDecorativeSfx) {
        let revision = Revision::new(8);
        let element = &page.text_elements[0];
        let label = "1:ABC123".to_owned();
        let crop = "crop-digest".to_owned();
        let mut crops = BTreeMap::new();
        crops.insert(
            element.id,
            SourceElementEvidence {
                ordinal: 1,
                source_debug_label: label.clone(),
                crop_blake3: crop.clone(),
            },
        );
        let mut evidence = PageSemanticEvidenceState::default();
        evidence.record(
            PageEvidenceKind::SourceDossier,
            PageEvidenceArtifact {
                revision,
                page_id: page.id,
                blake3: "source-dossier".to_owned(),
                element_crops: crops,
            },
        );
        evidence.record(
            PageEvidenceKind::SourceDebugArtifact,
            PageEvidenceArtifact {
                revision,
                page_id: page.id,
                blake3: "source-debug".to_owned(),
                element_crops: BTreeMap::new(),
            },
        );
        (
            evidence,
            ClassifyDecorativeSfx {
                page_ordinal: PageOrdinal::new(1),
                scene_revision: revision.get(),
                source_evidence_dossier_blake3: "source-dossier".to_owned(),
                source_debug_artifact_blake3: "source-debug".to_owned(),
                decisions: vec![DecorativeSfxClassification {
                    element: element.id.to_string(),
                    original_ordinal: 1,
                    disposition: DecorativeSfxDisposition::SkipDifficult,
                    source_crop_blake3: crop,
                    source_debug_label: label,
                    confidence: 0.96,
                    evidence: DecorativeSfxEvidence {
                        decorative_visual_form: "distorted display lettering follows impact lines"
                            .to_owned(),
                        sound_effect_page_function: "the marks convey an impact noise".to_owned(),
                        legibility_and_translation_value:
                            "overlapping irregular strokes obscure glyph identity and prevent reliable translation"
                                .to_owned(),
                        exclusion_of_dialogue_caption_ui_and_general_free_text:
                            "the crop is outside containers and carries no spoken or informational prose"
                                .to_owned(),
                    },
                    rationale: "All four independent source-pixel observations support difficult decorative SFX."
                        .to_owned(),
                }],
            },
        )
    }

    #[test]
    fn decorative_sfx_requires_complete_evidence_and_short_text_is_not_a_heuristic() {
        let page = sfx_classification_page(FREE_TEXT_ROLE, "ド");
        let (evidence, mut arguments) = sfx_classification_fixture(&page);
        assert!(
            validate_decorative_sfx_classifications(
                &page,
                &evidence,
                Revision::new(arguments.scene_revision),
                &arguments,
            )
            .is_ok()
        );
        arguments.decisions[0].disposition = DecorativeSfxDisposition::Translate;
        assert!(
            validate_decorative_sfx_classifications(
                &page,
                &evidence,
                Revision::new(arguments.scene_revision),
                &arguments,
            )
            .is_ok()
        );

        arguments.decisions[0]
            .evidence
            .exclusion_of_dialogue_caption_ui_and_general_free_text = String::new();
        assert!(
            validate_decorative_sfx_classifications(
                &page,
                &evidence,
                Revision::new(arguments.scene_revision),
                &arguments,
            )
            .is_err()
        );
    }

    #[test]
    fn decorative_sfx_classification_accepts_a_source_bound_detector_region() {
        let mut page = sfx_classification_page(FREE_TEXT_ROLE, "ド");
        let source_region = page.text_elements[0].source_region_id.unwrap();
        page.text_elements[0]
            .text_safe_region
            .as_mut()
            .unwrap()
            .association = Some(TargetRegionAssociation {
            layout_relation: TargetLayoutRelation::FitsTo,
            source_inside_target_relation: false,
        });
        assert_eq!(
            page.text_elements[0].text_safe_region.as_ref().unwrap().id,
            source_region
        );
        assert!(!is_actual_container_bound(&page.text_elements[0]));
        assert!(requires_decorative_sfx_disposition(&page.text_elements[0]));
        let (evidence, arguments) = sfx_classification_fixture(&page);

        assert!(
            validate_decorative_sfx_classifications(
                &page,
                &evidence,
                Revision::new(arguments.scene_revision),
                &arguments,
            )
            .is_ok()
        );
    }

    #[test]
    fn same_id_with_different_region_kind_is_container_bound() {
        let mut page = sfx_classification_page(FREE_TEXT_ROLE, "ド");
        let element = &mut page.text_elements[0];
        let source_kind = element.source_region_kind.as_deref().unwrap();
        element.text_safe_region.as_mut().unwrap().kind = format!("{source_kind}.alternate");

        assert!(is_actual_container_bound(element));
        assert!(!requires_decorative_sfx_disposition(element));
    }

    #[test]
    fn caption_and_ui_remain_required_and_reject_decorative_sfx_classification() {
        for role in ["dev.koharu.text.caption", UI_TEXT_ROLE] {
            let page = sfx_classification_page(role, "required text");
            let element = &page.text_elements[0];
            assert!(element.required, "role {role}");
            assert!(!requires_decorative_sfx_disposition(element), "role {role}");

            let (evidence, arguments) = sfx_classification_fixture(&page);
            assert!(
                validate_decorative_sfx_classifications(
                    &page,
                    &evidence,
                    Revision::new(arguments.scene_revision),
                    &arguments,
                )
                .is_err(),
                "role {role}"
            );
        }
    }

    #[test]
    fn grouping_only_uncontained_dialogue_is_pending_sfx_but_external_container_is_not() {
        let mut page =
            sfx_classification_page(crate::free_dialogue::DIALOGUE_ROLE, "required text");
        page.text_elements[0].logical_dialogue_memberships =
            vec![LogicalDialogueMembershipInspection {
                group_id: EntityId::new(),
                primary_render_element_id: EntityId::new(),
                target_region_id: EntityId::new(),
                member_ordinal: 1,
            }];
        let grouping_only = page.text_elements[0].id;
        let mut container_associated_element = page.text_elements[0].clone();
        container_associated_element.id = EntityId::new();
        container_associated_element.content_id = EntityId::new();
        container_associated_element
            .text_safe_region
            .as_mut()
            .unwrap()
            .id = EntityId::new();
        container_associated_element
            .text_safe_region
            .as_mut()
            .unwrap()
            .association = Some(TargetRegionAssociation {
            layout_relation: TargetLayoutRelation::FlowsIn,
            source_inside_target_relation: true,
        });
        let container_associated = container_associated_element.id;
        page.text_elements.push(container_associated_element);

        let inspection = ProjectInspection {
            project: ProjectState {
                kind: "isolated_disposable",
                revision: Revision::new(8),
                pages: vec![page],
            },
            configuration: HarnessConfiguration {
                source_language: "ja-JP".to_owned(),
                target_language: "ko-KR".to_owned(),
                ocr_model: "test".to_owned(),
                required_output_directory: String::new(),
                required_export_format: "png".to_owned(),
                quality_thresholds: QualityThresholds::default(),
                review_bundle_directory: String::new(),
                external_visual_judge_configured: false,
            },
            repair_history: RepairHistory {
                completed_review_attempts: 0,
                attempted_actions: Vec::new(),
                active_deterministic_plan: None,
                stop_diagnostic: None,
                correction_actions: Vec::new(),
            },
        };

        let candidates = decorative_sfx_disposition_candidates(&inspection);
        assert!(candidates.contains(&grouping_only));
        assert!(!candidates.contains(&container_associated));
    }

    #[test]
    fn pipeline_and_export_calls_reject_model_owned_arguments() {
        assert!(serde_json::from_str::<RunPipeline>("{}").is_ok());
        assert!(serde_json::from_str::<RunPipeline>(r#"{"pages":[]}"#).is_err());
        assert!(serde_json::from_str::<ExportPages>("{}").is_ok());
        assert!(
            serde_json::from_str::<ExportPages>(r#"{"format":"png","directory":"/tmp"}"#).is_err()
        );
    }

    fn pipeline_page(
        label: &str,
        stage_status: StageTelemetryStatus,
        semantic_after: StageSemanticCounts,
    ) -> PagePipelineTelemetry {
        PagePipelineTelemetry {
            page_id: EntityId::new(),
            label: label.to_owned(),
            stages: vec![StageTelemetry {
                stage: Stage::Translation,
                status: stage_status,
                model: Some("test-translator".to_owned()),
                elapsed_ms: Some(1),
            }],
            semantic_after,
            diagnosis: Vec::new(),
        }
    }

    #[test]
    fn source_analysis_then_pipeline_completes_with_valid_no_op_and_required_translation() {
        let skipped = pipeline_page(
            "difficult SFX",
            StageTelemetryStatus::NoOp,
            StageSemanticCounts {
                detected_text_elements: 1,
                skipped_difficult_sfx: 1,
                source_text_present: 0,
                ..Default::default()
            },
        );
        let translated = pipeline_page(
            "required dialogue",
            StageTelemetryStatus::Finished,
            StageSemanticCounts {
                detected_text_elements: 1,
                required_source_elements: 1,
                required_render_units: 1,
                source_text_present: 1,
                source_text_nonempty: 1,
                translation_present: 1,
                translation_nonempty: 1,
                target_language_translations: 1,
                render_eligible_translations: 1,
                visible_translations: 1,
                ..Default::default()
            },
        );
        let completed = AtomicBool::new(false);

        mark_pipeline_completed(&completed, &[skipped, translated]).unwrap();

        assert!(completed.load(Ordering::Acquire));
        assert_eq!(
            serde_json::to_value(StageTelemetryStatus::NoOp).unwrap(),
            json!("no_op")
        );
    }

    #[test]
    fn missing_required_translation_rejects_completion_without_setting_lifecycle_flag() {
        let missing = pipeline_page(
            "required dialogue",
            StageTelemetryStatus::NoOp,
            StageSemanticCounts {
                detected_text_elements: 1,
                required_source_elements: 1,
                required_render_units: 1,
                source_text_present: 1,
                source_text_nonempty: 1,
                ..Default::default()
            },
        );
        let completed = AtomicBool::new(false);

        let error = mark_pipeline_completed(&completed, &[missing]).unwrap_err();

        assert!(error.to_string().contains("0/0/0"));
        assert!(!completed.load(Ordering::Acquire));
    }

    #[test]
    fn english_manga_uses_the_multilingual_manga_ocr_profile() {
        let mut config = koharu_pipeline::PipelineConfig::default();
        configure_pipeline_for_harness(&mut config, Language::English, Language::Korean);
        assert!(matches!(config.ocr, OcrModel::HayaiOcr));
        assert_eq!(config.translation.target_language, Language::Korean);
    }

    #[test]
    fn revision_requires_a_reason_and_at_least_one_nonempty_change() {
        let empty = ReviseElement {
            element: EntityId::new().to_string(),
            source_text: None,
            translation_text: None,
            typography: None,
            layout: None,
            reason: "observed OCR corruption".to_owned(),
        };
        assert!(validate_revision_request(&empty).is_err());

        let blank_text = ReviseElement {
            source_text: Some("  ".to_owned()),
            ..empty
        };
        assert!(validate_revision_request(&blank_text).is_err());
    }

    #[test]
    fn page_semantic_batch_rejects_empty_wrong_language_and_layout_fields() {
        assert!(
            serde_json::from_value::<RevisePageTranslation>(json!({
                "page_ordinal": 1,
                "edits": [],
                "page_rationale": "missing evidence"
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<RevisePageTranslation>(json!({
                "page_ordinal": 1,
                "evidence": {
                    "scene_revision": 1,
                    "dossier_blake3": "dossier"
                },
                "edits": [],
                "page_rationale": "incomplete evidence"
            }))
            .is_err()
        );
        let empty = RevisePageTranslation {
            page_ordinal: PageOrdinal::new(1),
            evidence: PageTranslationEvidenceReference {
                scene_revision: 1,
                source_evidence_dossier_blake3: "source-dossier".to_owned(),
                source_debug_artifact_blake3: "source-debug".to_owned(),
                dossier_blake3: "dossier".to_owned(),
                debug_artifact_blake3: "debug".to_owned(),
            },
            edits: Vec::new(),
            page_rationale: "coherent page dialogue".to_owned(),
        };
        assert!(
            validate_page_translation_revision(&empty, Language::Japanese, Language::Korean)
                .is_err()
        );

        let wrong_language = RevisePageTranslation {
            edits: vec![PageTranslationEdit {
                element: EntityId::new().to_string(),
                source: None,
                translation: Some(LanguageTextEdit {
                    text: "자연스러운 대사".to_owned(),
                    language: "ja".to_owned(),
                }),
            }],
            ..empty
        };
        assert!(
            validate_page_translation_revision(
                &wrong_language,
                Language::Japanese,
                Language::Korean
            )
            .is_err()
        );
        assert!(
            serde_json::from_value::<RevisePageTranslation>(json!({
                "page_ordinal": 1,
                "evidence": {
                    "scene_revision": 1,
                    "source_evidence_dossier_blake3": "source-dossier",
                    "source_debug_artifact_blake3": "source-debug",
                    "dossier_blake3": "dossier",
                    "debug_artifact_blake3": "debug"
                },
                "edits": [{
                    "element": EntityId::new().to_string(),
                    "translation": { "text": "대사", "language": "ko" },
                    "layout": { "padding": 4 }
                }],
                "page_rationale": "page coherence"
            }))
            .is_err()
        );
    }

    #[tokio::test]
    async fn page_semantic_batch_rejects_cross_page_targets() {
        let mut session = Session::memory().await.unwrap();
        let patch = session
            .snapshot()
            .patch(|edit| {
                edit.add_page(PageDraft::new("first", 100.0, 100.0), At::End)?;
                edit.add_page(PageDraft::new("second", 100.0, 100.0), At::End)?;
                Ok(())
            })
            .unwrap();
        session.commit(patch).await.unwrap();
        let snapshot = session.snapshot();
        let pages = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
        assert!(validate_page_translation_targets(&snapshot, pages[0], &[pages[1]]).is_err());
        assert!(validate_page_translation_targets(&snapshot, pages[0], &[pages[0]]).is_ok());
    }

    #[test]
    fn active_page_clipping_plan_blocks_page_semantic_batch() {
        let element = EntityId::new();
        let page = EntityId::new();
        let revision = Revision::new(4);
        let plan = deterministic_repair_plan(
            &[crate::acceptance::AcceptanceRejection {
                code: AcceptanceRejectionCode::RenderedTextOutsidePage,
                page_id: page,
                element_id: Some(element),
                related_element_id: None,
                expected: json!({ "maximum_px": 0.5 }),
                actual: json!({ "value_px": 2.0 }),
            }],
            revision,
            None,
        );
        let mut review = review(VisualReviewStatus::PendingAgentReview, false, revision);
        review.deterministic_repair_plan = plan.clone();
        let edit = PageTranslationEdit {
            element: element.to_string(),
            source: None,
            translation: Some(LanguageTextEdit {
                text: "수정".to_owned(),
                language: "ko".to_owned(),
            }),
        };
        assert!(
            page_translation_evidence(
                Some(&review),
                &page_semantic_evidence(revision, page),
                revision,
                revision,
                page,
                "source-dossier-digest",
                "source-debug-digest",
                "dossier-digest",
                "debug-digest",
            )
            .is_ok()
        );
        assert!(
            validate_page_semantic_plan(plan.blocking_failures.first(), &[(element, &edit)])
                .is_err()
        );
    }

    fn compact_layout_failure(
        revision: Revision,
        page: EntityId,
        group: EntityId,
        primary: EntityId,
        secondary: EntityId,
    ) -> DeterministicRepairFailure {
        DeterministicRepairFailure {
            element_id: Some(primary),
            logical_group_id: Some(group),
            primary_render_element_id: Some(primary),
            member_ordinal_ids: vec![
                crate::repair::RepairLogicalDialogueMember {
                    ordinal: 1,
                    element_id: primary,
                    source_region_id: EntityId::new(),
                },
                crate::repair::RepairLogicalDialogueMember {
                    ordinal: 2,
                    element_id: secondary,
                    source_region_id: EntityId::new(),
                },
            ],
            code: AcceptanceRejectionCode::RenderedFontSizeBelowMinimum,
            expected: json!({ "minimum_px": 12.0, "page": page }),
            actual: json!({ "value_px": 10.8867 }),
            allowed_repair_fields: vec![RepairField::Typography, RepairField::Layout],
            next_action: Some(compact_translation_next_action(primary, revision)),
            required_evidence_revision: revision,
        }
    }

    #[test]
    fn oversized_logical_dialogue_blocks_general_semantic_revision_during_layout_plan() {
        let revision = Revision::new(8);
        let page = EntityId::new();
        let group = EntityId::new();
        let primary = EntityId::new();
        let secondary = EntityId::new();
        let failure = compact_layout_failure(revision, page, group, primary, secondary);
        let ordinary_semantic_edit = PageTranslationEdit {
            element: primary.to_string(),
            source: None,
            translation: Some(LanguageTextEdit {
                text: "짧게 고친 대사".to_owned(),
                language: "ko-KR".to_owned(),
            }),
        };

        assert!(
            validate_page_semantic_plan(Some(&failure), &[(primary, &ordinary_semantic_edit)])
                .is_err()
        );
        assert!(validate_compact_translation_plan(&failure, primary, group).is_ok());
        assert!(validate_compact_translation_plan(&failure, secondary, group).is_err());
        assert!(validate_compact_translation_plan(&failure, primary, EntityId::new()).is_err());
    }

    fn page_semantic_evidence(revision: Revision, page: EntityId) -> PageSemanticEvidenceState {
        PageSemanticEvidenceState {
            pages: BTreeMap::from([(
                page,
                PageSemanticEvidence {
                    source_dossier: Some(PageEvidenceArtifact {
                        revision,
                        page_id: page,
                        blake3: "source-dossier-digest".to_owned(),
                        element_crops: BTreeMap::new(),
                    }),
                    source_debug_artifact: Some(PageEvidenceArtifact {
                        revision,
                        page_id: page,
                        blake3: "source-debug-digest".to_owned(),
                        element_crops: BTreeMap::new(),
                    }),
                    dossier: Some(PageEvidenceArtifact {
                        revision,
                        page_id: page,
                        blake3: "dossier-digest".to_owned(),
                        element_crops: BTreeMap::new(),
                    }),
                    debug_artifact: Some(PageEvidenceArtifact {
                        revision,
                        page_id: page,
                        blake3: "debug-digest".to_owned(),
                        element_crops: BTreeMap::new(),
                    }),
                },
            )]),
        }
    }

    #[test]
    fn observing_two_pages_retains_addressable_evidence_for_the_first_page() {
        let revision = Revision::new(9);
        let page_a = EntityId::new();
        let page_b = EntityId::new();
        let mut evidence = PageSemanticEvidenceState::default();

        for page in [page_a, page_b] {
            for (kind, digest) in [
                (PageEvidenceKind::SourceDossier, "source-dossier"),
                (PageEvidenceKind::SourceDebugArtifact, "source-debug"),
                (PageEvidenceKind::Dossier, "translated-dossier"),
                (PageEvidenceKind::DebugArtifact, "translated-debug"),
            ] {
                evidence.record(
                    kind,
                    PageEvidenceArtifact {
                        revision,
                        page_id: page,
                        blake3: format!("{page}-{digest}"),
                        element_crops: BTreeMap::new(),
                    },
                );
            }
        }

        assert_eq!(
            complete_page_evidence(&evidence, page_a),
            Some((revision, page_a))
        );
        assert_eq!(
            complete_page_evidence(&evidence, page_b),
            Some((revision, page_b))
        );
        let expected_digest = format!("{page_a}-translated-dossier");
        assert_eq!(
            evidence
                .page(page_a)
                .and_then(|page| page.dossier.as_ref())
                .map(|artifact| artifact.blake3.as_str()),
            Some(expected_digest.as_str())
        );
    }

    fn page_semantic_evidence_with_members(
        revision: Revision,
        page: EntityId,
        members: &[crate::repair::RepairLogicalDialogueMember],
    ) -> PageSemanticEvidenceState {
        let mut evidence = page_semantic_evidence(revision, page);
        evidence
            .pages
            .get_mut(&page)
            .unwrap()
            .source_dossier
            .as_mut()
            .unwrap()
            .element_crops = members
            .iter()
            .map(|member| {
                (
                    member.element_id,
                    SourceElementEvidence {
                        ordinal: member.ordinal as usize,
                        source_debug_label: format!("{}-member", member.ordinal),
                        crop_blake3: format!("crop-{}", member.ordinal),
                    },
                )
            })
            .collect();
        evidence
    }

    fn agent_review_decision(accepted: bool) -> VisualReviewDecision {
        VisualReviewDecision {
            accepted,
            summary: if accepted {
                "source pixels and translated rendering agree across all required categories"
                    .to_owned()
            } else {
                "the translated rendering omits visible source dialogue".to_owned()
            },
            issues: if accepted {
                Vec::new()
            } else {
                vec![
                    "ordinal 2 is visible in the source but absent from the translation".to_owned(),
                ]
            },
            judgments: crate::review::RequiredJudgments {
                source_text_accurate: true,
                translation_meaning_accurate: true,
                target_language_natural: true,
                reading_order_preserved: true,
                content_complete_without_duplicates: accepted,
                typography_layout_acceptable: true,
                skipped_items_are_difficult_sfx_and_no_required_content_skipped: true,
            },
        }
    }

    fn agent_review_arguments(
        revision: Revision,
        _page: EntityId,
        decision: VisualReviewDecision,
    ) -> SubmitVisualSemanticReview {
        SubmitVisualSemanticReview {
            page_ordinal: PageOrdinal::new(1),
            scene_revision: revision.get(),
            source_evidence_dossier_blake3: "source-dossier-digest".to_owned(),
            source_debug_artifact_blake3: "source-debug-digest".to_owned(),
            dossier_blake3: "dossier-digest".to_owned(),
            debug_artifact_blake3: "debug-digest".to_owned(),
            compacted_translation_reviews: Vec::new(),
            decision,
        }
    }

    fn pending_agent_review(revision: Revision, page: EntityId) -> VisualReviewRecord {
        let mut record = review(VisualReviewStatus::PendingAgentReview, true, revision);
        record.judge.required = false;
        record.bundle.pages = vec![crate::review::ReviewBundlePage {
            page_id: page,
            label: "page.png".to_owned(),
            original: review_artifact("original"),
            rendered_preview: review_artifact("rendered"),
            semantic_elements: review_artifact("semantic"),
        }];
        record
    }

    fn review_artifact(digest: &str) -> crate::review::ReviewArtifact {
        crate::review::ReviewArtifact {
            path: String::new(),
            media_type: "application/test".to_owned(),
            byte_length: 1,
            blake3: digest.to_owned(),
        }
    }

    fn page_review_state(review: &VisualReviewRecord) -> PageReviewState {
        let mut state = PageReviewState::at_revision(
            review.bundle.pages.iter().map(|page| page.page_id),
            review.scene_revision,
        );
        for submitted in review.agent_reviews.clone() {
            state.record_review(submitted);
        }
        state
    }

    #[test]
    fn accepted_in_loop_review_with_fresh_bindings_satisfies_export_review_precondition() {
        let revision = Revision::new(9);
        let page = EntityId::new();
        let mut review = pending_agent_review(revision, page);
        let arguments = agent_review_arguments(revision, page, agent_review_decision(true));

        let submitted = validate_agent_visual_semantic_review(
            &review,
            &page_semantic_evidence(revision, page),
            revision,
            page,
            &[],
            &arguments,
        )
        .unwrap();
        record_agent_visual_semantic_review(&mut review, submitted);

        assert_eq!(review.status, VisualReviewStatus::Accepted);
        assert!(review.accepted());
        assert!(
            validate_visual_review_export_precondition(
                &review,
                &page_review_state(&review),
                revision,
            )
            .is_ok()
        );
        assert_eq!(review.agent_reviews.len(), 1);
    }

    #[test]
    fn page_b_mutation_keeps_page_a_review_current_and_stales_only_page_b() {
        let reviewed_revision = Revision::new(9);
        let mutated_revision = Revision::new(10);
        let page_a = EntityId::new();
        let page_b = EntityId::new();
        let mut evidence = page_semantic_evidence(reviewed_revision, page_a);
        evidence
            .pages
            .extend(page_semantic_evidence(reviewed_revision, page_b).pages);
        let mut review = pending_agent_review(reviewed_revision, page_a);
        review.bundle.pages.push(crate::review::ReviewBundlePage {
            page_id: page_b,
            label: "page-b.png".to_owned(),
            original: review_artifact("original-b"),
            rendered_preview: review_artifact("rendered-b"),
            semantic_elements: review_artifact("semantic-b"),
        });

        for page in [page_a, page_b] {
            let submitted = validate_agent_visual_semantic_review(
                &review,
                &evidence,
                reviewed_revision,
                page,
                &[],
                &agent_review_arguments(reviewed_revision, page, agent_review_decision(true)),
            )
            .unwrap();
            record_agent_visual_semantic_review(&mut review, submitted);
        }
        assert!(review.accepted());

        let mut page_reviews = PageReviewState::at_revision([page_a, page_b], reviewed_revision);
        for submitted in review.agent_reviews.clone() {
            page_reviews.record_review(submitted);
        }
        page_reviews.record_mutation(page_b, mutated_revision);

        assert!(page_reviews.current_review(page_a).is_some());
        assert!(page_reviews.current_review(page_b).is_none());
        assert!(page_reviews.evidence_is_current(&evidence, page_a));
        assert!(!page_reviews.evidence_is_current(&evidence, page_b));
    }

    #[test]
    fn two_page_review_recovery_requires_fresh_page_b_before_project_export() {
        let reviewed_revision = Revision::new(9);
        let mutated_revision = Revision::new(10);
        let page_a = EntityId::new();
        let page_b = EntityId::new();
        let mut evidence = page_semantic_evidence(reviewed_revision, page_a);
        evidence
            .pages
            .extend(page_semantic_evidence(reviewed_revision, page_b).pages);
        let mut review = pending_agent_review(reviewed_revision, page_a);
        review.bundle.pages.push(crate::review::ReviewBundlePage {
            page_id: page_b,
            label: "page-b.png".to_owned(),
            original: review_artifact("original-b"),
            rendered_preview: review_artifact("rendered-b"),
            semantic_elements: review_artifact("semantic-b"),
        });
        let mut page_reviews = PageReviewState::at_revision([page_a, page_b], reviewed_revision);

        let page_a_review = validate_agent_visual_semantic_review(
            &review,
            &evidence,
            reviewed_revision,
            page_a,
            &[],
            &agent_review_arguments(reviewed_revision, page_a, agent_review_decision(true)),
        )
        .unwrap();
        page_reviews.record_review(page_a_review.clone());
        record_agent_visual_semantic_review(&mut review, page_a_review);

        assert!(!review.accepted());
        assert!(
            validate_visual_review_export_precondition(&review, &page_reviews, reviewed_revision,)
                .is_err()
        );

        page_reviews.record_mutation(page_b, mutated_revision);
        review.scene_revision = mutated_revision;
        review.deterministic_repair_plan = deterministic_repair_plan(&[], mutated_revision, None);
        review.agent_reviews =
            page_reviews.current_reviews(review.bundle.pages.iter().map(|page| page.page_id));
        refresh_agent_visual_semantic_review_status(&mut review);

        assert!(page_reviews.current_review(page_a).is_some());
        assert!(page_reviews.evidence_is_current(&evidence, page_a));
        assert!(page_reviews.current_review(page_b).is_none());
        assert!(!page_reviews.evidence_is_current(&evidence, page_b));
        assert!(!review.accepted());

        evidence
            .pages
            .extend(page_semantic_evidence(mutated_revision, page_b).pages);
        let page_b_review = validate_agent_visual_semantic_review(
            &review,
            &evidence,
            mutated_revision,
            page_b,
            &[],
            &agent_review_arguments(mutated_revision, page_b, agent_review_decision(true)),
        )
        .unwrap();
        page_reviews.record_review(page_b_review.clone());
        record_agent_visual_semantic_review(&mut review, page_b_review);

        assert!(page_reviews.current_review(page_a).is_some());
        assert!(page_reviews.evidence_is_current(&evidence, page_a));
        assert!(page_reviews.current_review(page_b).is_some());
        assert!(page_reviews.evidence_is_current(&evidence, page_b));
        assert!(review.accepted());
        validate_visual_review_export_precondition(&review, &page_reviews, mutated_revision)
            .unwrap();
    }

    #[tokio::test]
    async fn two_page_semantic_review_lifecycle_advances_to_host_owned_export() {
        let fixture = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        let review_bundle = tempfile::tempdir().unwrap();
        let inputs = [
            ("page-b.png", image::Rgba([1, 2, 3, 255])),
            ("page-a.png", image::Rgba([4, 5, 6, 255])),
        ]
        .map(|(name, pixel)| {
            let input = fixture.path().join(name);
            let mut bytes = Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(2, 3, pixel))
                .write_to(&mut bytes, image::ImageFormat::Png)
                .unwrap();
            std::fs::write(&input, bytes.into_inner()).unwrap();
            input
        });
        let host = HarnessHost::create(
            inputs.to_vec(),
            Language::Japanese,
            Language::Korean,
            output.path().to_owned(),
            OutputFormat::Png,
            QualityThresholds::default(),
            review_bundle.path().to_owned(),
            None,
        )
        .await
        .unwrap();
        let snapshot = host.project.session().lock().await.snapshot();
        let page_ids = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
        let revision = snapshot.revision();
        let mut review = pending_agent_review(revision, page_ids[0]);
        review.bundle.pages.push(crate::review::ReviewBundlePage {
            page_id: page_ids[1],
            label: "page-a.png".to_owned(),
            original: review_artifact("original-a"),
            rendered_preview: review_artifact("rendered-a"),
            semantic_elements: review_artifact("semantic-a"),
        });
        host.source_analysis_completed
            .store(true, Ordering::Release);
        host.pipeline_completed.store(true, Ordering::Release);
        *host.page_reviews.lock() =
            PageReviewState::at_revision(page_ids.iter().copied(), revision);
        *host.visual_review.lock() = Some(review.clone());
        host.review_history.lock().push(review);

        fn assert_obligation_tool(
            host: &HarnessHost,
            expected_tool: &str,
            page_ordinal: usize,
            page_id: EntityId,
            stage: &str,
        ) {
            let tools = host.tools();
            assert_eq!(
                tools
                    .iter()
                    .map(|tool| tool.name.as_str())
                    .collect::<Vec<_>>(),
                [expected_tool]
            );
            let tool = &tools[0];
            assert_eq!(
                tool.parameters["properties"]["page_ordinal"]["const"],
                page_ordinal
            );
            assert!(tool.description.contains(&format!(
                "page ordinal {page_ordinal}, page ID {}, required stage {stage}",
                page_id
            )));
        }

        fn assert_next_obligation(
            invocation: &Invocation,
            page_ordinal: usize,
            page_id: EntityId,
            stage: &str,
        ) {
            assert_eq!(
                invocation.value["semantic_review_obligation"],
                json!({
                    "page_ordinal": page_ordinal,
                    "page_id": page_id,
                    "required_stage": stage,
                })
            );
        }

        async fn evidence_for(
            host: &HarnessHost,
            page_ordinal: usize,
            page_id: EntityId,
            predecessor: Option<EntityId>,
        ) -> [String; 4] {
            assert_obligation_tool(
                host,
                "inspect_page_evidence",
                page_ordinal,
                page_id,
                "inspect_page_evidence",
            );
            let evidence_before_submit = host.page_semantic_evidence.lock().clone();
            let premature_submit = host
                .invoke(
                    ToolCall {
                        call_id: format!("premature-submit-page-{page_ordinal}"),
                        name: "submit_visual_semantic_review".to_owned(),
                        arguments: json!({ "page_ordinal": page_ordinal }).to_string(),
                    },
                    &Control::default(),
                )
                .await
                .unwrap_err();
            assert!(
                premature_submit
                    .to_string()
                    .contains("submit_visual_semantic_review")
            );
            assert_eq!(*host.page_semantic_evidence.lock(), evidence_before_submit);

            let result = host
                .invoke(
                    ToolCall {
                        call_id: format!("page-{page_ordinal}-evidence"),
                        name: "inspect_page_evidence".to_owned(),
                        arguments: json!({ "page_ordinal": page_ordinal }).to_string(),
                    },
                    &Control::default(),
                )
                .await
                .unwrap();
            assert_next_obligation(
                &result,
                page_ordinal,
                page_id,
                "submit_visual_semantic_review",
            );
            for artifact in [
                "source_evidence",
                "source_debug_artifact",
                "translated_dossier",
                "rendered_debug_artifact",
            ] {
                assert!(result.value[artifact].is_object(), "missing {artifact}");
            }
            if let Some(predecessor) = predecessor {
                assert_eq!(
                    result.value["translated_dossier"]["accepted_predecessor_context"][0]["page_id"],
                    predecessor.to_string()
                );
                assert_eq!(
                    result.value["translated_dossier"]["accepted_predecessor_context"][0]["page_ordinal"],
                    1
                );
                assert_eq!(
                    result.value["translated_dossier"]["accepted_predecessor_context"][0]["review_accepted"],
                    true
                );
                assert!(result.value["translated_dossier"]["accepted_predecessor_context"][0]
                    ["source_translation_pairs"]
                    .is_array());
            }
            [
                "source_evidence_dossier_blake3",
                "source_debug_artifact_blake3",
                "dossier_blake3",
                "debug_artifact_blake3",
            ]
            .map(|field| result.value[field].as_str().unwrap().to_owned())
        }

        async fn accept_page(
            host: &HarnessHost,
            page_ordinal: usize,
            page_id: EntityId,
            revision: Revision,
            digests: &[String; 4],
        ) -> Invocation {
            assert_obligation_tool(
                host,
                "submit_visual_semantic_review",
                page_ordinal,
                page_id,
                "submit_visual_semantic_review",
            );
            let submit = host.tools().into_iter().next().unwrap();
            for (property, expected) in [
                ("page_ordinal", json!(page_ordinal)),
                ("scene_revision", json!(revision.get())),
                ("source_evidence_dossier_blake3", json!(digests[0])),
                ("source_debug_artifact_blake3", json!(digests[1])),
                ("dossier_blake3", json!(digests[2])),
                ("debug_artifact_blake3", json!(digests[3])),
            ] {
                assert_eq!(
                    submit.parameters["properties"][property]["const"], expected,
                    "submit schema must bind exact {property}"
                );
            }
            host.invoke(
                ToolCall {
                    call_id: format!("accept-page-{page_ordinal}"),
                    name: "submit_visual_semantic_review".to_owned(),
                    arguments: json!({
                        "page_ordinal": page_ordinal,
                        "scene_revision": revision,
                        "source_evidence_dossier_blake3": digests[0],
                        "source_debug_artifact_blake3": digests[1],
                        "dossier_blake3": digests[2],
                        "debug_artifact_blake3": digests[3],
                        "compacted_translation_reviews": [],
                        "decision": agent_review_decision(true),
                    })
                    .to_string(),
                },
                &Control::default(),
            )
            .await
            .unwrap()
        }

        let page_one_digests = evidence_for(&host, 1, page_ids[0], None).await;
        let accepted_page_one =
            accept_page(&host, 1, page_ids[0], revision, &page_one_digests).await;
        assert_next_obligation(&accepted_page_one, 2, page_ids[1], "inspect_page_evidence");

        assert_obligation_tool(
            &host,
            "inspect_page_evidence",
            2,
            page_ids[1],
            "inspect_page_evidence",
        );
        let evidence_before_wrong_page = host.page_semantic_evidence.lock().clone();
        let wrong_page = host
            .invoke(
                ToolCall {
                    call_id: "wrong-page-source".to_owned(),
                    name: "inspect_page_evidence".to_owned(),
                    arguments: json!({ "page_ordinal": 1 }).to_string(),
                },
                &Control::default(),
            )
            .await
            .unwrap_err();
        assert!(wrong_page.to_string().contains(&format!(
            "page ordinal 2, page ID {}, required stage inspect_page_evidence",
            page_ids[1]
        )));
        assert_eq!(
            *host.page_semantic_evidence.lock(),
            evidence_before_wrong_page
        );

        match host.completion().await.unwrap() {
            HostCompletion::Continue {
                phase,
                exposed_tools,
                reason,
                progress_marker,
            } => {
                assert_eq!(phase, "page_evidence");
                assert_eq!(exposed_tools, ["inspect_page_evidence"]);
                assert!(reason.contains(&format!(
                    "page ordinal 2, page ID {}, required stage inspect_page_evidence",
                    page_ids[1]
                )));
                let marker: Value = serde_json::from_str(&progress_marker).unwrap();
                assert_eq!(marker["page_ordinal"], 2);
                assert_eq!(marker["page_id"], page_ids[1].to_string());
                assert_eq!(marker["required_stage"], "inspect_page_evidence");
            }
            HostCompletion::Completed => panic!("pending page review must continue"),
        }

        let page_two_digests = evidence_for(&host, 2, page_ids[1], Some(page_ids[0])).await;
        let accepted_page_two =
            accept_page(&host, 2, page_ids[1], revision, &page_two_digests).await;
        assert!(accepted_page_two.value["semantic_review_obligation"].is_null());
        assert_eq!(
            host.tools()
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            ["export_pages"]
        );
        let review = host.visual_review_record().unwrap();
        validate_visual_review_export_precondition(&review, &host.page_reviews.lock(), revision)
            .unwrap();

        assert!(serde_json::from_value::<ExportPages>(json!({})).is_ok());
        assert!(
            serde_json::from_value::<ExportPages>(json!({
                "format": "psd",
                "directory": "/model/chosen"
            }))
            .is_err()
        );
        let paths = export_pages(
            host.renderer.clone(),
            host.rasterizer().await.unwrap(),
            snapshot,
            Vec::new(),
            host.output_format.into(),
            host.output_directory.clone(),
        )
        .await
        .unwrap();

        assert_eq!(
            paths,
            [
                output.path().join("0001_page-b.png"),
                output.path().join("0002_page-a.png"),
            ]
        );
        assert!(paths.iter().all(|path| path.is_file()));
    }

    #[tokio::test]
    async fn page_b_model_context_includes_page_a_source_and_translation_from_inspection() {
        let fixture = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        let review_bundle = tempfile::tempdir().unwrap();
        let inputs = ["page-a.png", "page-b.png"].map(|name| {
            let input = fixture.path().join(name);
            let mut bytes = Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
                32,
                32,
                image::Rgba([255, 255, 255, 255]),
            ))
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
            std::fs::write(&input, bytes.into_inner()).unwrap();
            input
        });
        let host = HarnessHost::create(
            inputs.to_vec(),
            Language::Japanese,
            Language::Korean,
            output.path().to_owned(),
            OutputFormat::Png,
            QualityThresholds::default(),
            review_bundle.path().to_owned(),
            None,
        )
        .await
        .unwrap();
        let mut session = host.project.session().lock().await;
        let snapshot = session.snapshot();
        let pages = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
        let page_a = pages[0];
        let page_b = pages[1];
        let patch = snapshot
            .patch(|edit| {
                let content = edit.add_text_content(page_a, At::End)?;
                let layer = edit.add_text_layer(
                    page_a,
                    At::End,
                    content,
                    &TextLayout {
                        origin: koharu_scene::Origin::User,
                        kind: TextLayoutKind::Paragraph,
                    },
                )?;
                edit.set(
                    content,
                    &SourceText {
                        text: Authored::user("前ページの情報".to_owned()),
                        language: Some(LanguageTag::new("ja-JP")?),
                    },
                )?;
                edit.set(
                    content,
                    &Translation {
                        text: Authored::user("앞 페이지 정보".to_owned()),
                        language: Some(LanguageTag::new("ko-KR")?),
                    },
                )?;
                edit.set(
                    layer,
                    &Typography {
                        origin: koharu_scene::Origin::User,
                        preferred_font: None,
                        font_weight: None,
                        font_style: None,
                        size: Some(12.0),
                        auto_fit: true,
                        color: Some([0, 0, 0, 255]),
                        stroke_color: None,
                        stroke_width: None,
                        alignment: None,
                        writing_mode: Some(WritingMode::Horizontal),
                        extensions: Default::default(),
                    },
                )?;
                edit.set(layer, &Geometry::rectangle(2.0, 2.0, 28.0, 28.0))?;
                Ok(())
            })
            .unwrap();
        let revision = session.commit(patch).await.unwrap().snapshot.revision();
        drop(session);
        *host.current_semantic_evidence_page.lock() = Some(PageTarget {
            ordinal: PageOrdinal::new(2),
            id: page_b,
        });
        host.page_semantic_evidence
            .lock()
            .pages
            .extend(page_semantic_evidence(revision, page_b).pages);

        let context = Host::context(&host).await.unwrap();

        assert_eq!(context["project"]["pages"][0]["page_ordinal"], 1);
        assert_eq!(context["project"]["pages"][1]["page_ordinal"], 2);
        assert_eq!(context["project"]["pages"][0]["id"], page_a.to_string());
        assert_eq!(context["project"]["pages"][1]["id"], page_b.to_string());
        assert_eq!(
            context["project"]["pages"][0]["text_elements"][0]["source"]["text"],
            "前ページの情報"
        );
        assert_eq!(
            context["project"]["pages"][0]["text_elements"][0]["translation"]["text"],
            "앞 페이지 정보"
        );
    }

    #[test]
    fn export_requires_a_current_semantic_review_for_every_page() {
        let reviewed_revision = Revision::new(9);
        let current_revision = Revision::new(10);
        let page_a = EntityId::new();
        let page_b = EntityId::new();
        let mut evidence = page_semantic_evidence(reviewed_revision, page_a);
        evidence
            .pages
            .extend(page_semantic_evidence(reviewed_revision, page_b).pages);
        let mut review = pending_agent_review(reviewed_revision, page_a);
        review.bundle.pages.push(crate::review::ReviewBundlePage {
            page_id: page_b,
            label: "page-b.png".to_owned(),
            original: review_artifact("original-b"),
            rendered_preview: review_artifact("rendered-b"),
            semantic_elements: review_artifact("semantic-b"),
        });
        let mut page_reviews = PageReviewState::at_revision([page_a, page_b], reviewed_revision);
        for page in [page_a, page_b] {
            let submitted = validate_agent_visual_semantic_review(
                &review,
                &evidence,
                reviewed_revision,
                page,
                &[],
                &agent_review_arguments(reviewed_revision, page, agent_review_decision(true)),
            )
            .unwrap();
            page_reviews.record_review(submitted.clone());
            record_agent_visual_semantic_review(&mut review, submitted);
        }

        page_reviews.record_mutation(page_b, current_revision);
        review.scene_revision = current_revision;
        review.agent_reviews =
            page_reviews.current_reviews(review.bundle.pages.iter().map(|page| page.page_id));
        refresh_agent_visual_semantic_review_status(&mut review);

        assert!(
            validate_visual_review_export_precondition(&review, &page_reviews, current_revision,)
                .is_err()
        );
        assert_eq!(review.agent_reviews.len(), 1);
        assert_eq!(review.agent_reviews[0].page_id, page_a);
    }

    #[test]
    fn in_loop_review_rejects_a_false_difficult_sfx_classification() {
        let revision = Revision::new(10);
        let page = EntityId::new();
        let mut review = pending_agent_review(revision, page);
        let mut decision = agent_review_decision(true);
        decision.accepted = false;
        decision.summary = "ordinal 1 was incorrectly skipped as SFX".to_owned();
        decision.issues =
            vec!["ordinal 1 is required caption text, not difficult decorative SFX".to_owned()];
        decision
            .judgments
            .skipped_items_are_difficult_sfx_and_no_required_content_skipped = false;
        let arguments = agent_review_arguments(revision, page, decision);

        let submitted = validate_agent_visual_semantic_review(
            &review,
            &page_semantic_evidence(revision, page),
            revision,
            page,
            &[],
            &arguments,
        )
        .unwrap();
        record_agent_visual_semantic_review(&mut review, submitted);

        assert_eq!(review.status, VisualReviewStatus::Rejected);
        assert!(!review.accepted());
        assert!(
            validate_visual_review_export_precondition(
                &review,
                &page_review_state(&review),
                revision,
            )
            .is_err()
        );
    }

    #[test]
    fn in_loop_review_binding_failures_and_false_acceptance_do_not_mutate_review() {
        let revision = Revision::new(9);
        let page = EntityId::new();
        let review = pending_agent_review(revision, page);
        let original = serde_json::to_value(&review).unwrap();

        let missing = PageSemanticEvidenceState::default();
        let arguments = agent_review_arguments(revision, page, agent_review_decision(true));
        assert!(
            validate_agent_visual_semantic_review(
                &review,
                &missing,
                revision,
                page,
                &[],
                &arguments,
            )
            .is_err()
        );

        let stale_arguments = agent_review_arguments(
            Revision::new(revision.get() - 1),
            page,
            agent_review_decision(true),
        );
        assert!(
            validate_agent_visual_semantic_review(
                &review,
                &page_semantic_evidence(revision, page),
                revision,
                page,
                &[],
                &stale_arguments,
            )
            .is_err()
        );

        let mut mismatched = agent_review_arguments(revision, page, agent_review_decision(true));
        mismatched.debug_artifact_blake3 = "wrong-debug-digest".to_owned();
        assert!(
            validate_agent_visual_semantic_review(
                &review,
                &page_semantic_evidence(revision, page),
                revision,
                page,
                &[],
                &mismatched,
            )
            .is_err()
        );

        let mut false_acceptance = agent_review_decision(true);
        false_acceptance.judgments.translation_meaning_accurate = false;
        let false_arguments = agent_review_arguments(revision, page, false_acceptance);
        assert!(
            validate_agent_visual_semantic_review(
                &review,
                &page_semantic_evidence(revision, page),
                revision,
                page,
                &[],
                &false_arguments,
            )
            .is_err()
        );
        assert_eq!(serde_json::to_value(&review).unwrap(), original);
    }

    #[test]
    fn command_judge_mode_cannot_be_bypassed_by_agent_submission() {
        let revision = Revision::new(9);
        let page = EntityId::new();
        let review = review(VisualReviewStatus::PendingAgentReview, true, revision);
        let arguments = agent_review_arguments(revision, page, agent_review_decision(true));

        assert!(
            validate_agent_visual_semantic_review(
                &review,
                &page_semantic_evidence(revision, page),
                revision,
                page,
                &[],
                &arguments,
            )
            .unwrap_err()
            .to_string()
            .contains("configured command judge")
        );
        assert!(!review.accepted());
        assert!(review.agent_reviews.is_empty());
    }

    #[test]
    fn in_loop_rejection_blocks_export_review_precondition_and_retains_issues() {
        let revision = Revision::new(9);
        let page = EntityId::new();
        let mut review = pending_agent_review(revision, page);
        let arguments = agent_review_arguments(revision, page, agent_review_decision(false));
        let submitted = validate_agent_visual_semantic_review(
            &review,
            &page_semantic_evidence(revision, page),
            revision,
            page,
            &[],
            &arguments,
        )
        .unwrap();

        record_agent_visual_semantic_review(&mut review, submitted);

        assert_eq!(review.status, VisualReviewStatus::Rejected);
        assert!(!review.accepted());
        assert!(
            validate_visual_review_export_precondition(
                &review,
                &page_review_state(&review),
                revision,
            )
            .is_err()
        );
        assert_eq!(
            review.agent_reviews[0].decision.issues,
            arguments.decision.issues
        );
    }

    #[test]
    fn in_loop_semantic_review_rejects_compacted_meaning_drift_against_member_crops() {
        let revision = Revision::new(9);
        let page = EntityId::new();
        let group = EntityId::new();
        let primary = EntityId::new();
        let secondary = EntityId::new();
        let failure = compact_layout_failure(revision, page, group, primary, secondary);
        let requirement = CompactTranslationReviewRequirement {
            logical_group_id: group,
            primary_render_element_id: primary,
            members: failure.member_ordinal_ids.clone(),
        };
        let evidence =
            page_semantic_evidence_with_members(revision, page, &failure.member_ordinal_ids);
        let issue = "압축 대사가 두 번째 원문의 '기쁘다' 의미를 누락했다".to_owned();
        let mut decision = agent_review_decision(true);
        decision.accepted = false;
        decision.summary = "압축 과정에서 원문 의미가 손실됐다".to_owned();
        decision.issues = vec![issue.clone()];
        decision.judgments.translation_meaning_accurate = false;
        let mut arguments = agent_review_arguments(revision, page, decision);
        arguments.compacted_translation_reviews = vec![CompactedTranslationSemanticReview {
            logical_group: group.to_string(),
            primary_render_element: primary.to_string(),
            member_evidence: compact_member_evidence(
                &evidence,
                page,
                &failure.member_ordinal_ids,
                "test",
            )
            .unwrap(),
            source_fidelity_preserved: false,
            target_language_natural: true,
            rationale: "두 원문 크롭을 순서대로 대조했다".to_owned(),
            issues: vec![issue.clone()],
        }];
        let mut review = pending_agent_review(revision, page);
        let submitted = validate_agent_visual_semantic_review(
            &review,
            &evidence,
            revision,
            page,
            &[requirement],
            &arguments,
        )
        .unwrap();
        record_agent_visual_semantic_review(&mut review, submitted);

        assert_eq!(review.status, VisualReviewStatus::Rejected);
        assert_eq!(review.agent_reviews[0].decision.issues, vec![issue]);
        assert!(
            validate_visual_review_export_precondition(
                &review,
                &page_review_state(&review),
                revision,
            )
            .is_err()
        );
    }

    #[test]
    fn fresh_four_part_evidence_authorizes_revision_and_binds_all_digests() {
        let revision = Revision::new(6);
        let page = EntityId::new();
        let evidence = page_translation_evidence(
            None,
            &page_semantic_evidence(revision, page),
            revision,
            revision,
            page,
            "source-dossier-digest",
            "source-debug-digest",
            "dossier-digest",
            "debug-digest",
        )
        .unwrap();

        assert_eq!(
            evidence,
            RevisionEvidence::PageTranslationVisualEvidence {
                revision,
                page_id: page,
                source_evidence_dossier_blake3: "source-dossier-digest".to_owned(),
                source_debug_artifact_blake3: "source-debug-digest".to_owned(),
                dossier_blake3: "dossier-digest".to_owned(),
                debug_artifact_blake3: "debug-digest".to_owned(),
            }
        );
        let trace_evidence = serde_json::to_value(&evidence).unwrap();
        assert_eq!(
            trace_evidence["source_evidence_dossier_blake3"],
            "source-dossier-digest"
        );
        assert_eq!(
            trace_evidence["source_debug_artifact_blake3"],
            "source-debug-digest"
        );
        assert_eq!(trace_evidence["dossier_blake3"], "dossier-digest");
        assert_eq!(trace_evidence["debug_artifact_blake3"], "debug-digest");
    }

    #[test]
    fn page_semantic_batch_rejects_missing_stale_mismatched_source_evidence_without_mutation() {
        let revision = Revision::new(6);
        let page = EntityId::new();
        let other_page = EntityId::new();

        assert!(
            page_translation_evidence(
                None,
                &PageSemanticEvidenceState::default(),
                revision,
                revision,
                page,
                "source-dossier-digest",
                "source-debug-digest",
                "dossier-digest",
                "debug-digest",
            )
            .is_err()
        );
        assert!(
            page_translation_evidence(
                None,
                &page_semantic_evidence(Revision::new(5), page),
                revision,
                revision,
                page,
                "source-dossier-digest",
                "source-debug-digest",
                "dossier-digest",
                "debug-digest",
            )
            .is_err()
        );
        let mut wrong_page = page_semantic_evidence(revision, page);
        wrong_page
            .pages
            .get_mut(&page)
            .unwrap()
            .source_debug_artifact
            .as_mut()
            .unwrap()
            .page_id = other_page;
        let unchanged = wrong_page.clone();
        assert!(
            page_translation_evidence(
                None,
                &wrong_page,
                revision,
                revision,
                page,
                "source-dossier-digest",
                "source-debug-digest",
                "dossier-digest",
                "debug-digest",
            )
            .is_err()
        );
        assert_eq!(wrong_page, unchanged);
        assert!(
            page_translation_evidence(
                None,
                &page_semantic_evidence(revision, page),
                revision,
                revision,
                page,
                "wrong-source-dossier",
                "source-debug-digest",
                "dossier-digest",
                "debug-digest",
            )
            .is_err()
        );
        assert!(
            page_translation_evidence(
                None,
                &page_semantic_evidence(revision, page),
                revision,
                revision,
                page,
                "source-dossier-digest",
                "wrong-source-debug",
                "dossier-digest",
                "debug-digest",
            )
            .is_err()
        );
    }

    #[test]
    fn accepted_external_review_rejects_even_fresh_page_semantic_evidence() {
        let revision = Revision::new(7);
        let page = EntityId::new();
        let review = review(VisualReviewStatus::Accepted, true, revision);

        assert!(
            page_translation_evidence(
                Some(&review),
                &page_semantic_evidence(revision, page),
                revision,
                revision,
                page,
                "source-dossier-digest",
                "source-debug-digest",
                "dossier-digest",
                "debug-digest",
            )
            .is_err()
        );
    }

    #[test]
    fn revision_rejects_non_finite_or_empty_layout_bounds() {
        let request = ReviseElement {
            element: EntityId::new().to_string(),
            source_text: None,
            translation_text: None,
            typography: None,
            layout: Some(RepairLayout {
                kind: Some(RepairLayoutKind::Paragraph),
                bounds: Some(RepairBounds {
                    x: 0.0,
                    y: 0.0,
                    width: 0.0,
                    height: 20.0,
                }),
                padding: None,
            }),
            reason: "text frame does not contain the rendered line".to_owned(),
        };
        assert!(validate_revision_request(&request).is_err());
    }

    #[test]
    fn revision_accepts_per_edge_safe_region_padding() {
        let request = serde_json::from_value::<ReviseElement>(json!({
            "element": EntityId::new().to_string(),
            "layout": {
                "padding": { "top": 8.0, "right": 14.0, "bottom": 8.0, "left": 20.0 }
            },
            "reason": "glyph ink crosses the curved left balloon wall"
        }))
        .unwrap();
        assert!(validate_revision_request(&request).is_ok());
        assert_eq!(request.layout.unwrap().padding.unwrap().left, 20.0);
    }

    #[test]
    fn text_safe_repair_schema_rejects_null_malformed_and_nonpositive_deltas() {
        let preview_schema = &tool_definitions()
            .iter()
            .find(|tool| tool.name == "preview_increase_text_safe_padding")
            .unwrap()
            .parameters;
        assert_eq!(preview_schema["additionalProperties"], false);
        let required = preview_schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("element")));
        assert!(required.contains(&json!("inset_delta_px")));
        assert!(required.contains(&json!("reason")));
        let inset_schema = &preview_schema["$defs"]["TextSafeInsetDelta"];
        assert_eq!(inset_schema["additionalProperties"], false);
        let required_edges = inset_schema["required"].as_array().unwrap();
        assert_eq!(required_edges.len(), 4);
        for edge in ["top", "right", "bottom", "left"] {
            assert!(required_edges.contains(&json!(edge)));
            assert_eq!(inset_schema["properties"][edge]["type"], "number");
        }

        let element = EntityId::new().to_string();
        assert!(
            serde_json::from_value::<PreviewTextSafeLayoutRepair>(json!({
                "element": element,
                "inset_delta_px": null,
                "reason": "increase glyph clearance"
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<PreviewTextSafeLayoutRepair>(json!({
                "element": EntityId::new().to_string(),
                "inset_delta_px": {
                    "top": null,
                    "right": 1.0,
                    "bottom": 1.0,
                    "left": 1.0
                },
                "reason": "increase glyph clearance"
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<PreviewTextSafeLayoutRepair>(json!({
                "element": EntityId::new().to_string(),
                "layout": { "bounds": null },
                "reason": "ambiguous raw geometry"
            }))
            .is_err()
        );
        for invalid in [-1.0, 0.0, f64::INFINITY, f64::NAN] {
            assert!(
                validate_text_safe_inset_delta(TextSafeInsetDelta {
                    top: invalid,
                    right: 1.0,
                    bottom: 1.0,
                    left: 1.0,
                })
                .is_err()
            );
        }
    }

    #[test]
    fn positive_padding_delta_only_shrinks_layout_geometry() {
        let before = ElementBounds {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 60.0,
        };
        let candidate = inset_layout_bounds(
            before,
            TextSafeInsetDelta {
                top: 2.0,
                right: 3.0,
                bottom: 4.0,
                left: 5.0,
            },
        )
        .unwrap();
        assert_eq!(candidate.x, 15.0);
        assert_eq!(candidate.y, 22.0);
        assert_eq!(candidate.width, 92.0);
        assert_eq!(candidate.height, 54.0);
        assert!(
            validate_nonexpanding_layout_bounds(
                before,
                ElementBounds {
                    x: 9.0,
                    y: 20.0,
                    width: 101.0,
                    height: 60.0,
                },
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn compact_translation_rejects_unsafe_shape_nonshorter_and_missing_evidence_without_mutation()
     {
        assert!(
            serde_json::from_value::<PreviewCompactTranslation>(json!({
                "logical_group": EntityId::new().to_string(),
                "primary_render_element": EntityId::new().to_string(),
                "candidate_translation": { "text": "짧은 대사", "language": "ko-KR" },
                "evidence": {
                    "scene_revision": 8,
                    "source_evidence_dossier_blake3": "source-dossier-digest",
                    "source_debug_artifact_blake3": "source-debug-digest",
                    "dossier_blake3": "dossier-digest",
                    "debug_artifact_blake3": "debug-digest"
                },
                "rationale": "fit at the unchanged minimum",
                "layout": { "bounds": [0, 0, 1, 1] }
            }))
            .is_err()
        );
        assert!(validate_compact_translation_length("짧은 대사", "더 길어진 대사").is_err());
        assert!(validate_compact_translation_length("짧은 대사", "짧은 대사").is_err());

        let revision = Revision::new(8);
        let page = EntityId::new();
        let primary = EntityId::new();
        let secondary = EntityId::new();
        let group = EntityId::new();
        let failure = compact_layout_failure(revision, page, group, primary, secondary);
        let reference = PageTranslationEvidenceReference {
            scene_revision: revision.get(),
            source_evidence_dossier_blake3: "source-dossier-digest".to_owned(),
            source_debug_artifact_blake3: "source-debug-digest".to_owned(),
            dossier_blake3: "dossier-digest".to_owned(),
            debug_artifact_blake3: "debug-digest".to_owned(),
        };
        assert!(
            compact_translation_evidence(
                &PageSemanticEvidenceState::default(),
                revision,
                page,
                &reference,
                &failure.member_ordinal_ids,
            )
            .is_err()
        );

        let mut session = Session::memory().await.unwrap();
        let patch = session
            .snapshot()
            .patch(|edit| {
                edit.add_page(PageDraft::new("must not commit", 100.0, 100.0), At::End)?;
                Ok(())
            })
            .unwrap();
        let region = EntityId::new();
        assert!(
            commit_compact_translation_patch_if_safe(
                &mut session,
                patch,
                Revision::ZERO,
                &compact_metrics(10.8867, false, region),
                &compact_metrics(11.5, false, region),
                &QualityThresholds::default(),
            )
            .await
            .is_err()
        );
        assert_eq!(session.snapshot().revision(), Revision::ZERO);
        assert_eq!(session.snapshot().pages().len(), 0);
    }

    #[test]
    fn undersized_evidenced_logical_dialogue_does_not_enter_repair_routing() {
        let revision = Revision::new(8);
        let mut page = sfx_classification_page(crate::free_dialogue::DIALOGUE_ROLE, "長い台詞");
        let primary = page.text_elements[0].id;
        let primary_source = page.text_elements[0].source_region_id.unwrap();
        let group = EntityId::new();
        let target = EntityId::new();
        let mut secondary = page.text_elements[0].clone();
        secondary.id = EntityId::new();
        secondary.content_id = EntityId::new();
        secondary.source_region_id = Some(EntityId::new());
        secondary.source.as_mut().unwrap().text = "続き".to_owned();
        secondary.source.as_mut().unwrap().language = Some("ja-JP".to_owned());
        let secondary_id = secondary.id;
        let secondary_source = secondary.source_region_id.unwrap();
        let primary_element = &mut page.text_elements[0];
        primary_element.source.as_mut().unwrap().language = Some("ja-JP".to_owned());
        primary_element.translation = Some(SemanticText {
            text: "아주 길게 번역된 한국어 대사입니다".to_owned(),
            language: Some("ko-KR".to_owned()),
        });
        primary_element
            .text_safe_region
            .as_mut()
            .unwrap()
            .association = Some(TargetRegionAssociation {
            layout_relation: TargetLayoutRelation::FlowsIn,
            source_inside_target_relation: true,
        });
        primary_element.final_scene.font_size_px = Some(11.3796);
        page.text_elements.push(secondary);
        page.logical_dialogue_groups = vec![LogicalDialogueGroupInspection {
            group_id: group,
            primary_render_element_id: primary,
            target_region_id: target,
            logical_source_text: "長い台詞 続き".to_owned(),
            members: vec![
                LogicalDialogueMemberInspection {
                    ordinal: 1,
                    element_id: primary,
                    content_id: page.text_elements[0].content_id,
                    source_region_id: primary_source,
                    source_text: "長い台詞".to_owned(),
                },
                LogicalDialogueMemberInspection {
                    ordinal: 2,
                    element_id: secondary_id,
                    content_id: page.text_elements[1].content_id,
                    source_region_id: secondary_source,
                    source_text: "続き".to_owned(),
                },
            ],
        }];
        let inspection = ProjectInspection {
            project: ProjectState {
                kind: "isolated_disposable",
                revision,
                pages: vec![page],
            },
            configuration: HarnessConfiguration {
                source_language: "ja-JP".to_owned(),
                target_language: "ko-KR".to_owned(),
                ocr_model: "test".to_owned(),
                required_output_directory: String::new(),
                required_export_format: "png".to_owned(),
                quality_thresholds: QualityThresholds::default(),
                review_bundle_directory: String::new(),
                external_visual_judge_configured: false,
            },
            repair_history: RepairHistory {
                completed_review_attempts: 1,
                attempted_actions: Vec::new(),
                active_deterministic_plan: None,
                stop_diagnostic: None,
                correction_actions: Vec::new(),
            },
        };
        let plan = deterministic_repair_plan(
            &[AcceptanceRejection {
                code: AcceptanceRejectionCode::RenderedFontSizeBelowMinimum,
                page_id: inspection.project.pages[0].id,
                element_id: Some(primary),
                related_element_id: None,
                expected: json!({ "minimum_px": 12.0 }),
                actual: json!({ "value_px": 11.3796 }),
            }],
            revision,
            Some(&inspection),
        );
        assert!(plan.blocking_failures.is_empty());
        assert_eq!(plan.unresolved_failure_count, 0);
        assert!(validate_compact_translation_length("긴 한국어 대사", "긴 한국어 대사").is_err());
        let invalid = compact_metrics(11.5, false, target);
        assert!(
            validate_compact_translation_candidate(
                &compact_metrics(11.3796, false, target),
                &invalid,
                &QualityThresholds::default(),
            )
            .is_err()
        );
    }

    #[test]
    fn tiny_required_dialogue_without_approved_anchor_retains_assessment_without_repair_route() {
        let revision = Revision::new(9);
        let mut page = sfx_classification_page(FREE_TEXT_ROLE, "え～");
        let element = &mut page.text_elements[0];
        element.source.as_mut().unwrap().language = Some("ja-JP".to_owned());
        element.translation = Some(SemanticText {
            text: "어~".to_owned(),
            language: Some("ko-KR".to_owned()),
        });
        element.final_scene.layout_bounds =
            element.source_geometry.as_ref().map(|value| value.bounds);
        element.final_scene.font_size_px = Some(9.0);
        element.free_dialogue_anchor_assessment = Some(assess_free_dialogue_anchor(
            &image::GrayImage::from_pixel(100, 100, image::Luma([0])),
            FreeDialogueAnchorInput {
                source_region_id: element.source_region_id.unwrap(),
                source_bounds: element.source_geometry.as_ref().unwrap().bounds,
                source_text: "え～",
                target_text: "어~",
                source_scene_role: FREE_TEXT_ROLE,
                required: true,
                has_finite_verified_container_relation: false,
                source_language: Some("ja-JP"),
                target_language: Some("ko-KR"),
                source_writing_mode: Some(WritingMode::Vertical),
                target_writing_mode: Some(WritingMode::Horizontal),
                target_ink_luma: 0,
                other_source_bounds: &[],
            },
        ));
        let assessment = element
            .free_dialogue_anchor_assessment
            .as_ref()
            .unwrap()
            .clone();
        assert!(assessment.role_gate.eligible_uncontained_source);
        assert_eq!(
            assessment.role_gate.classified_role,
            "required_uncontained_compact_text"
        );
        assert!(assessment.selected_candidate.is_none());
        let element_id = element.id;
        let source_region_id = element.source_region_id.unwrap();
        let inspection = ProjectInspection {
            project: ProjectState {
                kind: "isolated_disposable",
                revision,
                pages: vec![page],
            },
            configuration: HarnessConfiguration {
                source_language: "ja-JP".to_owned(),
                target_language: "ko-KR".to_owned(),
                ocr_model: "test".to_owned(),
                required_output_directory: String::new(),
                required_export_format: "png".to_owned(),
                quality_thresholds: QualityThresholds::default(),
                review_bundle_directory: String::new(),
                external_visual_judge_configured: false,
            },
            repair_history: RepairHistory {
                completed_review_attempts: 1,
                attempted_actions: Vec::new(),
                active_deterministic_plan: None,
                stop_diagnostic: None,
                correction_actions: Vec::new(),
            },
        };
        let plan = deterministic_repair_plan(
            &[
                AcceptanceRejection {
                    code: AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior,
                    page_id: inspection.project.pages[0].id,
                    element_id: Some(element_id),
                    related_element_id: None,
                    expected: json!({
                        "minimum_clearance_px": 4.0,
                        "target_layout_anchor": {
                            "kind": "source_region",
                            "target_writing_mode": "Horizontal"
                        }
                    }),
                    actual: json!({
                        "region_id": source_region_id,
                        "minimum_clearance_px": 0.5
                    }),
                },
                AcceptanceRejection {
                    code: AcceptanceRejectionCode::RenderedFontSizeBelowMinimum,
                    page_id: inspection.project.pages[0].id,
                    element_id: Some(element_id),
                    related_element_id: None,
                    expected: json!({ "minimum_px": 12.0 }),
                    actual: json!({ "value_px": 9.0 }),
                },
            ],
            revision,
            Some(&inspection),
        );
        assert!(plan.blocking_failures.is_empty());
        assert!(plan.terminal_diagnostic.is_none());
        let failed_anchor = serde_json::to_value(
            crate::free_dialogue::failed_free_dialogue_anchor_evidence(&assessment).unwrap(),
        )
        .unwrap();
        assert_eq!(failed_anchor["role_gate"]["source_text"], "え～");
        assert!(failed_anchor["candidate_count"].as_u64().unwrap() > 0);
        assert!(failed_anchor["source_bound_fallback"].is_null());
        assert!(
            failed_anchor["terminal_reason"]
                .as_str()
                .unwrap()
                .contains("original source bounds cannot geometrically contain")
        );
        assert!(
            failed_anchor["rejection_reason_counts"]
                ["high_edge_density_or_character_face_outline_structure"]
                .as_u64()
                .unwrap()
                > 0
        );
    }

    #[test]
    fn fitting_source_geometry_records_native_fallback_without_repair_route() {
        let revision = Revision::new(9);
        let mut page = sfx_classification_page(FREE_TEXT_ROLE, "ほう");
        let element = &mut page.text_elements[0];
        let bounds = ElementBounds {
            x: 673.572_784_315_471_2,
            y: 870.776_877_422_549_4,
            width: 25.846_618_869_057_693,
            height: 49.071_245_154_901_135,
        };
        let geometry = ElementGeometry {
            points: vec![
                ElementPoint {
                    x: 679.691_199_575_786,
                    y: bounds.y,
                },
                ElementPoint {
                    x: bounds.x + bounds.width,
                    y: 873.374_144_700_553_2,
                },
                ElementPoint {
                    x: 693.300_987_924_214,
                    y: bounds.y + bounds.height,
                },
                ElementPoint {
                    x: bounds.x,
                    y: 917.250_855_299_446_8,
                },
            ],
            bounds,
        };
        assert!(geometry.points[0].y < geometry.points[1].y);
        let source_region_id = element.source_region_id.unwrap();
        element.source_geometry = Some(geometry.clone());
        element.text_safe_region = Some(TextSafeRegion {
            id: source_region_id,
            kind: "dev.koharu.region.text".to_owned(),
            geometry,
            association: None,
        });
        element.source.as_mut().unwrap().language = Some("ja-JP".to_owned());
        element.source_writing_mode = Some(WritingMode::Vertical);
        element.translation = Some(SemanticText {
            text: "허.".to_owned(),
            language: Some("ko-KR".to_owned()),
        });
        element.typography = Some(Typography {
            origin: koharu_scene::Origin::User,
            preferred_font: None,
            font_weight: None,
            font_style: None,
            size: Some(9.0),
            auto_fit: true,
            color: Some([0, 0, 0, 255]),
            stroke_color: None,
            stroke_width: None,
            alignment: None,
            writing_mode: Some(WritingMode::Horizontal),
            extensions: Default::default(),
        });
        element.final_scene.layout_bounds = Some(bounds);
        element.final_scene.font_size_px = Some(9.0);
        element.free_dialogue_anchor_assessment = Some(assess_free_dialogue_anchor(
            &image::GrayImage::from_pixel(849, 1_200, image::Luma([0])),
            FreeDialogueAnchorInput {
                source_region_id,
                source_bounds: bounds,
                source_text: "ほう",
                target_text: "허.",
                source_scene_role: FREE_TEXT_ROLE,
                required: true,
                has_finite_verified_container_relation: false,
                source_language: Some("ja-JP"),
                target_language: Some("ko-KR"),
                source_writing_mode: Some(WritingMode::Vertical),
                target_writing_mode: Some(WritingMode::Horizontal),
                target_ink_luma: 0,
                other_source_bounds: &[],
            },
        ));
        assert!(
            element
                .free_dialogue_anchor_assessment
                .as_ref()
                .unwrap()
                .source_bound_fallback
                .is_some()
        );
        let element_id = element.id;
        let inspection = ProjectInspection {
            project: ProjectState {
                kind: "isolated_disposable",
                revision,
                pages: vec![page],
            },
            configuration: HarnessConfiguration {
                source_language: "ja-JP".to_owned(),
                target_language: "ko-KR".to_owned(),
                ocr_model: "test".to_owned(),
                required_output_directory: String::new(),
                required_export_format: "png".to_owned(),
                quality_thresholds: QualityThresholds::default(),
                review_bundle_directory: String::new(),
                external_visual_judge_configured: false,
            },
            repair_history: RepairHistory {
                completed_review_attempts: 1,
                attempted_actions: Vec::new(),
                active_deterministic_plan: None,
                stop_diagnostic: None,
                correction_actions: Vec::new(),
            },
        };
        let plan = deterministic_repair_plan(
            &[AcceptanceRejection {
                code: AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior,
                page_id: inspection.project.pages[0].id,
                element_id: Some(element_id),
                related_element_id: None,
                expected: json!({
                    "minimum_clearance_px": 4.0,
                    "target_layout_anchor": {
                        "kind": "source_region",
                        "target_writing_mode": "Horizontal"
                    }
                }),
                actual: json!({
                    "region_id": source_region_id,
                    "minimum_clearance_px": 0.5
                }),
            }],
            revision,
            Some(&inspection),
        );

        assert!(plan.terminal_diagnostic.is_none());
        assert!(plan.blocking_failures.is_empty());
    }

    #[test]
    fn serialized_fallback_contract_invokes_its_exact_documented_payload() {
        let element_id = EntityId::new();
        let mut tool = tool_definition("preview_text_layout").clone();
        constrain_repair_tool(
            &mut tool,
            &RepairToolConstraint {
                name: "preview_text_layout",
                operation: Some("controlled_source_bound_vertical_interjection_fallback"),
                element: Some(element_id),
                logical_group: None,
                page: None,
                revision: Some(Revision::new(9)),
                preview_id: None,
                allowed_fields: vec![RepairField::Layout],
            },
        );
        let exposed = serde_json::to_value(tool).unwrap();
        assert!(
            exposed["description"]
                .as_str()
                .unwrap()
                .contains("exposed only after a recorded no-adjacent-candidate-passed outcome")
        );
        let options_schema = &exposed["parameters"]["$defs"]["TextLayoutOptions"];
        assert_eq!(options_schema["anyOf"].as_array().unwrap().len(), 1);
        assert!(
            options_schema["anyOf"][0]["$ref"]
                .as_str()
                .unwrap()
                .ends_with("SourceBoundInterjectionFallbackOptions")
        );
        let fallback_schema =
            &exposed["parameters"]["$defs"]["SourceBoundInterjectionFallbackOptions"];
        assert_eq!(fallback_schema["additionalProperties"], false);
        assert_eq!(
            fallback_schema["required"],
            json!(["source_bound_interjection_fallback"])
        );
        assert_eq!(fallback_schema["properties"].as_object().unwrap().len(), 1);

        let element = element_id.to_string();
        let documented_payload =
            NATIVE_VERTICAL_FALLBACK_PAYLOAD_EXAMPLE.replace("<ELEMENT_ID>", &element);
        let call = ToolCall {
            call_id: "documented-native-vertical-preview".to_owned(),
            name: "preview_text_layout".to_owned(),
            arguments: documented_payload,
        };
        let preview_request = preview_text_layout_arguments(&call).unwrap();
        assert_eq!(
            apply_line_break_policy(
                "허.",
                preview_request.options.line_break_policy(),
                preview_request.options.max_lines(),
            )
            .unwrap(),
            "허."
        );
        let preview_typography = source_bound_interjection_typography(
            Some(&Typography {
                origin: koharu_scene::Origin::Generated(Generation::new(
                    ProducerId::new("dev.koharu.test").unwrap(),
                )),
                preferred_font: None,
                font_weight: Some(700),
                font_style: None,
                size: Some(9.0),
                auto_fit: true,
                color: Some([0, 0, 0, 255]),
                stroke_color: None,
                stroke_width: None,
                alignment: None,
                writing_mode: Some(WritingMode::Horizontal),
                extensions: Default::default(),
            }),
            12.0,
        )
        .unwrap();
        assert_eq!(preview_typography.size, Some(12.0));
        assert!(!preview_typography.auto_fit);
        assert_eq!(preview_typography.alignment, Some(TextAlignment::Center));
        assert_eq!(preview_typography.writing_mode, Some(WritingMode::Vertical));

        for (modifier, value) in [
            ("line_break_policy", json!("preserve_existing")),
            ("max_lines", json!(1)),
            ("alignment", json!("center")),
            ("font_scale", json!(0.9)),
            (
                "safe_padding_increase_px",
                json!({ "top": 1.0, "right": 1.0, "bottom": 1.0, "left": 1.0 }),
            ),
        ] {
            let mut options = json!({
                "source_bound_interjection_fallback": "native_vertical"
            });
            options[modifier] = value;
            let invalid = ToolCall {
                call_id: format!("invalid-native-vertical-{modifier}"),
                name: "preview_text_layout".to_owned(),
                arguments: json!({
                    "element": element,
                    "options": options,
                    "reason": "must reject modifiers"
                })
                .to_string(),
            };
            assert!(
                preview_text_layout_arguments(&invalid).is_err(),
                "{modifier}"
            );
        }
    }

    fn text_safe_measurement(region_id: EntityId, clearance: f64) -> TextSafeContainment {
        TextSafeContainment {
            region_id,
            region_kind: "dev.koharu.region.bubble".to_owned(),
            region_bounds: ElementBounds {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 60.0,
            },
            required_padding_px: 4.0,
            minimum_clearance_px: Some(clearance),
            glyph_ink_pixels: 100,
            violation_pixels: usize::from(clearance < 4.0),
            violations: Vec::new(),
        }
    }

    fn text_safe_metrics(
        region_id: EntityId,
        clearance: f64,
        font_size: f64,
        rejection_codes: &[&str],
    ) -> TextLayoutPreviewMetrics {
        let mut metrics = layout_metrics(1, false, 0.0);
        metrics.deterministic_element_accepted = rejection_codes.is_empty();
        metrics.deterministic_rejection_codes = rejection_codes
            .iter()
            .map(|code| (*code).to_owned())
            .collect();
        metrics.rendered_font_size_px = Some(font_size);
        metrics.text_safe_clearance = Some(text_safe_measurement(region_id, clearance));
        let anchor = metrics.target_layout_anchor.as_mut().unwrap();
        anchor.rendered_area_coverage = Some(0.20);
        anchor.required_area_coverage = Some(0.08);
        metrics
    }

    #[test]
    fn clearance_evidence_does_not_authorize_a_repair_commit() {
        let revision = Revision::new(11);
        let element = EntityId::new();
        let plan = deterministic_repair_plan(
            &[AcceptanceRejection {
                code: AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior,
                page_id: EntityId::new(),
                element_id: Some(element),
                related_element_id: None,
                expected: json!({ "minimum_clearance_px": 4.0 }),
                actual: json!({ "minimum_clearance_px": 3.27 }),
            }],
            revision,
            None,
        );
        assert_eq!(plan.unresolved_failure_count, 0);
        assert!(plan.blocking_failures.is_empty());
    }

    #[tokio::test]
    async fn clearance_repair_is_atomic_for_global_failures_staleness_and_valid_increase() {
        let mut session = Session::memory().await.unwrap();
        let patch = session
            .snapshot()
            .patch(|edit| {
                edit.add_page(PageDraft::new("page", 100.0, 100.0), At::End)?;
                Ok(())
            })
            .unwrap();
        let region_id = EntityId::new();
        let before = text_safe_metrics(
            region_id,
            3.27,
            18.0,
            &["rendered_text_outside_text_safe_interior"],
        );
        let worse = text_safe_metrics(
            region_id,
            -7.99,
            18.0,
            &["rendered_text_outside_text_safe_interior"],
        );

        let error = commit_text_safe_patch_if_globally_safe(
            &mut session,
            patch.clone(),
            Revision::ZERO,
            &before,
            &worse,
            &QualityThresholds::default(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("rejected without mutation"));
        assert_eq!(session.snapshot().revision(), Revision::ZERO);
        assert_eq!(session.snapshot().pages().len(), 0);

        let below_minimum = text_safe_metrics(
            region_id,
            4.105,
            9.700_962,
            &["rendered_font_size_below_minimum"],
        );
        let error = commit_text_safe_patch_if_globally_safe(
            &mut session,
            patch.clone(),
            Revision::ZERO,
            &before,
            &below_minimum,
            &QualityThresholds::default(),
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("rendered_font_size_below_minimum")
        );
        assert_eq!(session.snapshot().revision(), Revision::ZERO);
        assert_eq!(session.snapshot().pages().len(), 0);

        let valid = text_safe_metrics(region_id, 4.25, 12.5, &[]);
        let revision = commit_text_safe_patch_if_globally_safe(
            &mut session,
            patch,
            Revision::ZERO,
            &before,
            &valid,
            &QualityThresholds::default(),
        )
        .await
        .unwrap();
        assert!(4.25 > 3.27);
        assert_eq!(revision, Revision::new(1));
        assert_eq!(session.snapshot().pages().len(), 1);

        let stale_patch = session
            .snapshot()
            .patch(|edit| {
                edit.add_page(PageDraft::new("stale", 100.0, 100.0), At::End)?;
                Ok(())
            })
            .unwrap();
        let error = commit_text_safe_patch_if_globally_safe(
            &mut session,
            stale_patch,
            Revision::ZERO,
            &before,
            &valid,
            &QualityThresholds::default(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("preview is stale"));
        assert_eq!(session.snapshot().revision(), Revision::new(1));
        assert_eq!(session.snapshot().pages().len(), 1);
    }

    fn layout_metrics(
        line_count: usize,
        overflow: bool,
        source_overflow: f64,
    ) -> TextLayoutPreviewMetrics {
        TextLayoutPreviewMetrics {
            deterministic_element_accepted: !overflow && source_overflow <= 2.0,
            deterministic_rejection_codes: if overflow {
                vec!["typesetting_failed".to_owned()]
            } else if source_overflow > 2.0 {
                vec!["rendered_text_outside_source_region".to_owned()]
            } else {
                Vec::new()
            },
            finite_positive_layout: true,
            line_count: Some(line_count),
            overflow,
            rendered_font_size_px: Some(18.0),
            rendered_glyph_height_px: Some(14.0),
            maximum_reasonable_line_count: 12,
            target_layout_anchor: Some(crate::acceptance::LayoutAnchorMeasurement {
                kind: crate::acceptance::LayoutAnchorKind::SourceRegion,
                region_id: None,
                region_kind: Some("dev.koharu.region.text".to_owned()),
                bounds: None,
                source_writing_mode: None,
                target_writing_mode: None,
                association_confidence: None,
                association_reason: "test_source_region_anchor".to_owned(),
                coverage_policy:
                    crate::acceptance::LayoutAnchorCoveragePolicy::ConfiguredSourceAnchorFloor,
                expected_text_units: 0,
                target_anchor_area_px2: None,
                expected_ink_area_px2: 0.0,
                measured_ink_bounds_area_px2: None,
                rendered_area_coverage: None,
                required_area_coverage: None,
                rendered_overflow_px: Some(source_overflow),
            }),
            page_overflow_px: Some(0.0),
            text_safe_clearance: None,
            renderer_diagnostics: if overflow {
                vec!["text_overflow".to_owned()]
            } else {
                Vec::new()
            },
        }
    }

    #[tokio::test]
    async fn controlled_line_layout_rejects_invalid_or_unsafe_without_mutation_and_commits_safe_preview()
     {
        let invalid = PreviewTextLayout {
            element: EntityId::new().to_string(),
            options: TextLayoutOptions::Controlled(ControlledTextLayoutOptions {
                line_break_policy: KoreanLineBreakPolicy::KoreanKeepWordsBalanced,
                max_lines: 0,
                alignment: None,
                font_scale: Some(1.0),
                safe_padding_increase_px: None,
            }),
            reason: "fit Korean dialogue".to_owned(),
        };
        assert!(validate_text_layout_request(&invalid).is_err());

        let mut session = Session::memory().await.unwrap();
        let patch = session
            .snapshot()
            .patch(|edit| {
                edit.add_page(PageDraft::new("page", 100.0, 100.0), At::End)?;
                Ok(())
            })
            .unwrap();
        let thresholds = QualityThresholds::default();
        let before = layout_metrics(4, true, 3.0);
        let unsafe_candidate = layout_metrics(4, true, 3.5);
        assert!(
            commit_text_layout_patch_if_safe(
                &mut session,
                patch.clone(),
                &before,
                &unsafe_candidate,
                3,
                &thresholds,
            )
            .await
            .is_err()
        );
        assert_eq!(session.snapshot().revision(), Revision::ZERO);
        assert_eq!(session.snapshot().pages().len(), 0);

        let safe_candidate = layout_metrics(3, false, 1.0);
        let revision = commit_text_layout_patch_if_safe(
            &mut session,
            patch,
            &before,
            &safe_candidate,
            3,
            &thresholds,
        )
        .await
        .unwrap();
        assert_eq!(revision, Revision::new(1));
        assert_eq!(session.snapshot().pages().len(), 1);
    }

    #[tokio::test]
    async fn verified_ui_panel_layout_target_is_exact_and_role_gated() {
        let mut session = Session::memory().await.unwrap();
        let mut ids = None;
        let generation = koharu_scene::Generation {
            producer: koharu_scene::ProducerId::new(RASTER_PANEL_DETECTOR).unwrap(),
            model: Some(RASTER_PANEL_DETECTOR_VERSION.to_owned()),
            confidence: None,
        };
        let patch = {
            let mut edit = session.snapshot().edit_as(generation.clone());
            (|| {
                let page = edit.add_page(PageDraft::new("page", 200.0, 100.0), At::End)?;
                let source = edit.add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(60.0, 35.0, 68.815, 18.75),
                    Some("text".to_owned()),
                )?;
                let panel = edit.add_analysis_region::<PanelRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(20.0, 15.0, 150.0, 60.0),
                    Some("panel".to_owned()),
                )?;
                edit.set(
                    panel,
                    &DetectionAnalysis {
                        origin: koharu_scene::Origin::Generated(generation.clone()),
                        labels: vec![koharu_scene::DetectionLabel {
                            kind: PanelRegion::kind(),
                            confidence: 0.96,
                        }],
                    },
                )?;
                let content = edit.add_text_content(page, At::End)?;
                let layer = edit.add_text_layer(
                    page,
                    At::End,
                    content,
                    &TextLayout {
                        origin: koharu_scene::Origin::User,
                        kind: TextLayoutKind::Paragraph,
                    },
                )?;
                edit.relate::<koharu_scene::RecognizedFrom>(content, source)?;
                edit.relate::<FitsTo>(layer, source)?;
                edit.set(
                    content,
                    &TextRole {
                        origin: koharu_scene::Origin::User,
                        role: UI_TEXT_ROLE.to_owned(),
                    },
                )?;
                ids = Some((page, source, panel, content, layer));
                Ok::<_, koharu_scene::Error>(())
            })()
            .unwrap();
            edit.finish().unwrap()
        };
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let (page, source, panel, content, layer) = ids.unwrap();
        let panel_candidate = DetectedPanelCandidate {
            region_id: panel,
            region_kind: PanelRegion::KIND.to_owned(),
            geometry: element_geometry(snapshot.component::<Geometry>(panel).unwrap().unwrap()),
            detection_label: "panel".to_owned(),
            detection_confidence: 0.96,
            detector_producer: RASTER_PANEL_DETECTOR.to_owned(),
            detector_model: Some(RASTER_PANEL_DETECTOR_VERSION.to_owned()),
            raster_evidence: crate::ui_panel::test_raster_panel_evidence(
                source,
                ElementBounds {
                    x: 60.0,
                    y: 35.0,
                    width: 68.815,
                    height: 18.75,
                },
                ElementBounds {
                    x: 20.0,
                    y: 15.0,
                    width: 150.0,
                    height: 60.0,
                },
            ),
        };
        let decision = UiPanelAnchorDecision {
            schema_version: 2,
            decision: "verified_required_ui_panel_text_anchor",
            evidence_revision: snapshot.revision(),
            decision_revision: snapshot.revision(),
            page_id: page,
            original_ordinal: 1,
            element_id: layer,
            content_id: content,
            source_region_id: source,
            source_ocr: SemanticText {
                text: "故障中".to_owned(),
                language: Some("ja-JP".to_owned()),
            },
            source_crop_blake3: "crop".to_owned(),
            source_debug_label: "1:UI".to_owned(),
            panel: panel_candidate,
            source_containment_ratio: 1.0,
            source_intersection_ratio: 0.07,
            classifier: UiPanelClassifier {
                kind: "agent_visual_semantic",
                configured_model: None,
                tool_call_id: "call".to_owned(),
            },
            evidence: UiPanelEvidence {
                ui_role_and_function: "device status".to_owned(),
                finite_visible_panel_or_screen: "visible rectangle".to_owned(),
                source_to_panel_relation: "contained".to_owned(),
                text_safe_interior: "clear interior".to_owned(),
                visual_or_detection_provenance: "detector plus original pixels".to_owned(),
            },
            confidence: 0.97,
            association_reason: "verified_required_ui_text_inside_explicit_detected_panel"
                .to_owned(),
        };

        assert_eq!(
            verified_ui_panel_layout_target(&snapshot, layer, content, Some(&decision)).unwrap(),
            Some(panel)
        );
        let free_text = snapshot
            .patch(|edit| {
                edit.set(
                    content,
                    &TextRole {
                        origin: koharu_scene::Origin::User,
                        role: FREE_TEXT_ROLE.to_owned(),
                    },
                )
            })
            .unwrap();
        let free_text = snapshot.preview([&free_text]).unwrap();
        assert!(
            verified_ui_panel_layout_target(&free_text, layer, content, Some(&decision)).is_err()
        );
        assert!(free_text.relation_from::<FlowsIn>(layer).unwrap().is_none());
    }

    #[test]
    fn exact_source_contained_ui_is_host_bound_and_never_routes_source_clearance() {
        let revision = Revision::new(4);
        let decision_revision = Revision::new(5);
        let mut page = sfx_classification_page(FREE_TEXT_ROLE, "故障中");
        page.width = 849.0;
        page.height = 1200.0;
        let exact_element = page.text_elements[0].id;
        let source_region = page.text_elements[0].source_region_id.unwrap();
        let source_bounds = ElementBounds {
            x: 181.573_242_187_5,
            y: 501.562_5,
            width: 68.815_429_687_5,
            height: 18.75,
        };
        let source_geometry = ElementGeometry {
            points: vec![
                ElementPoint {
                    x: source_bounds.x,
                    y: source_bounds.y,
                },
                ElementPoint {
                    x: source_bounds.x + source_bounds.width,
                    y: source_bounds.y,
                },
                ElementPoint {
                    x: source_bounds.x + source_bounds.width,
                    y: source_bounds.y + source_bounds.height,
                },
                ElementPoint {
                    x: source_bounds.x,
                    y: source_bounds.y + source_bounds.height,
                },
            ],
            bounds: source_bounds,
        };
        page.text_elements[0].source_geometry = Some(source_geometry.clone());
        page.text_elements[0].text_safe_region = Some(TextSafeRegion {
            id: source_region,
            kind: TextRegion::KIND.to_owned(),
            geometry: source_geometry,
            association: Some(TargetRegionAssociation {
                layout_relation: TargetLayoutRelation::FitsTo,
                source_inside_target_relation: false,
            }),
        });
        let mut unrelated = page.text_elements[0].clone();
        unrelated.id = EntityId::new();
        unrelated.content_id = EntityId::new();
        unrelated.source_region_id = Some(EntityId::new());
        unrelated.source.as_mut().unwrap().text = "別の文字".to_owned();
        page.text_elements.push(unrelated);

        let panel_id = EntityId::new();
        let panel_bounds = ElementBounds {
            x: 173.0,
            y: 485.0,
            width: 106.0,
            height: 46.0,
        };
        let panel_geometry = ElementGeometry {
            points: vec![
                ElementPoint { x: 173.0, y: 485.0 },
                ElementPoint { x: 279.0, y: 485.0 },
                ElementPoint { x: 279.0, y: 531.0 },
                ElementPoint { x: 173.0, y: 531.0 },
            ],
            bounds: panel_bounds,
        };
        let panel = DetectedPanelCandidate {
            region_id: panel_id,
            region_kind: PanelRegion::KIND.to_owned(),
            geometry: panel_geometry.clone(),
            detection_label: "source-raster-closed-ui-panel".to_owned(),
            detection_confidence: 0.98,
            detector_producer: RASTER_PANEL_DETECTOR.to_owned(),
            detector_model: Some(RASTER_PANEL_DETECTOR_VERSION.to_owned()),
            raster_evidence: crate::ui_panel::test_raster_panel_evidence(
                source_region,
                source_bounds,
                panel_bounds,
            ),
        };
        page.detected_panel_candidates.push(panel);
        let source_evidence = BTreeMap::from([(
            exact_element,
            SourceElementEvidence {
                ordinal: 10,
                source_debug_label: "10:UI".to_owned(),
                crop_blake3: "exact-crop".to_owned(),
            },
        )]);

        let decisions = prepare_deterministic_ui_panel_bindings(
            &page,
            &source_evidence,
            revision,
            decision_revision,
        )
        .unwrap();
        assert_eq!(decisions.len(), 1);
        let decision = decisions.into_iter().next().unwrap();
        assert_eq!(decision.element_id, exact_element);
        assert_eq!(decision.source_region_id, source_region);
        assert_eq!(decision.panel.region_id, panel_id);
        assert_eq!(
            decision.panel.raster_evidence.safe_interior_bbox,
            ElementBounds {
                x: 175.0,
                y: 487.0,
                width: 102.0,
                height: 42.0,
            }
        );

        let element = &mut page.text_elements[0];
        element.text_role = Some(UI_TEXT_ROLE.to_owned());
        element.verified_ui_panel_anchor = Some(decision);
        element.text_safe_region = Some(TextSafeRegion {
            id: panel_id,
            kind: PanelRegion::KIND.to_owned(),
            geometry: panel_geometry,
            association: None,
        });
        let inspection = ProjectInspection {
            project: ProjectState {
                kind: "isolated_disposable",
                revision: decision_revision,
                pages: vec![page],
            },
            configuration: HarnessConfiguration {
                source_language: "ja-JP".to_owned(),
                target_language: "ko-KR".to_owned(),
                ocr_model: "test".to_owned(),
                required_output_directory: String::new(),
                required_export_format: "png".to_owned(),
                quality_thresholds: QualityThresholds::default(),
                review_bundle_directory: String::new(),
                external_visual_judge_configured: false,
            },
            repair_history: RepairHistory {
                completed_review_attempts: 0,
                attempted_actions: Vec::new(),
                active_deterministic_plan: None,
                stop_diagnostic: None,
                correction_actions: Vec::new(),
            },
        };
        let plan = deterministic_repair_plan(
            &[AcceptanceRejection {
                code: AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior,
                page_id: inspection.project.pages[0].id,
                element_id: Some(exact_element),
                related_element_id: None,
                expected: json!({ "minimum_clearance_px": 4.0 }),
                actual: json!({ "minimum_clearance_px": 0.5 }),
            }],
            decision_revision,
            Some(&inspection),
        );
        assert!(plan.blocking_failures.is_empty());
        assert_eq!(plan.unresolved_failure_count, 0);
    }

    #[tokio::test]
    async fn exact_e_tilde_adjacent_preview_uses_text_safe_fits_to_relation() {
        let source_bounds = ElementBounds {
            x: 472.166_170_167_871_9,
            y: 805.619_672_014_744,
            width: 17.425_472_164_256_234,
            height: 38.760_655_970_512_06,
        };
        let candidate_bounds = ElementBounds {
            x: 460.522_538_208_935_94,
            y: 721.619_672_014_744,
            width: 32.0,
            height: 20.0,
        };
        let mut source_page = image::RgbaImage::from_pixel(
            849,
            1200,
            image::Rgba([u8::MAX, u8::MAX, u8::MAX, u8::MAX]),
        );
        let source_pixels = (806..844)
            .flat_map(|y| (473..489).map(move |x| (x, y)))
            .collect::<Vec<_>>();
        for &(x, y) in &source_pixels {
            source_page.put_pixel(x, y, image::Rgba([0, 0, 0, u8::MAX]));
        }
        let mut cleanup = image::RgbaImage::new(849, 1200);
        for &(x, y) in &source_pixels {
            cleanup.put_pixel(x, y, image::Rgba([u8::MAX, u8::MAX, u8::MAX, u8::MAX]));
        }
        let encode = |image: image::RgbaImage| {
            let mut bytes = Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(image)
                .write_to(&mut bytes, image::ImageFormat::Png)
                .unwrap();
            Arc::<[u8]>::from(bytes.into_inner())
        };
        let source_page = encode(source_page);
        let cleanup = encode(cleanup);
        let fixture = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        let review_bundle = tempfile::tempdir().unwrap();
        let input = fixture.path().join("page.png");
        std::fs::write(&input, source_page.as_ref()).unwrap();
        let host = HarnessHost::create(
            vec![input],
            Language::Japanese,
            Language::Korean,
            output.path().to_owned(),
            OutputFormat::Png,
            QualityThresholds::default(),
            review_bundle.path().to_owned(),
            None,
        )
        .await
        .unwrap();
        let raster = image::GrayImage::from_pixel(240, 180, image::Luma([244]));
        let assessment = assess_free_dialogue_anchor(
            &raster,
            FreeDialogueAnchorInput {
                source_region_id: EntityId::new(),
                source_bounds: ElementBounds {
                    x: 60.0,
                    y: 50.0,
                    width: 16.0,
                    height: 40.0,
                },
                source_text: "え～",
                target_text: "에~",
                source_scene_role: FREE_TEXT_ROLE,
                required: true,
                has_finite_verified_container_relation: false,
                source_language: Some("ja-JP"),
                target_language: Some("ko-KR"),
                source_writing_mode: Some(WritingMode::Vertical),
                target_writing_mode: Some(WritingMode::Horizontal),
                target_ink_luma: 0,
                other_source_bounds: &[],
            },
        );
        let mut selected = assessment.selected_candidate.clone().unwrap();
        assert!(assessment.rejection_reasons.is_empty());
        selected.bounds = candidate_bounds;
        selected.distance_px = 64.0;
        selected.association.source_center_distance_px = 93.481_889_139_785_76;
        selected.association.source_edge_distance_px = 64.0;
        selected.association.maximum_source_edge_distance_px = 66.140_983_955_768_09;
        selected.association.source_edge_distance_within_cap = true;
        selected.association.nearest_other_source_center_distance_px = None;
        selected.association.attribution_margin_px = None;
        selected.association.has_required_attribution_margin = true;
        selected.association.reading_order_preserved = true;
        selected.association.preserves_reading_order_and_attribution = true;
        let mut role_gate = assessment.role_gate.clone();
        role_gate.source_bounds = source_bounds;

        let mut session = host.project.session().lock().await;
        let generation = Generation {
            producer: ProducerId::new(RASTER_FREE_DIALOGUE_DETECTOR).unwrap(),
            model: Some(RASTER_FREE_DIALOGUE_DETECTOR_VERSION.to_owned()),
            confidence: None,
        };
        let mut ids = None;
        let setup = {
            let snapshot = session.snapshot();
            let mut edit = snapshot.edit_as(generation.clone());
            (|| {
                let page = snapshot.pages().next().unwrap().id();
                let cleanup_layer = edit.add_entity(page, At::Start)?;
                edit.set(
                    cleanup_layer,
                    &koharu_scene::RasterLayer {
                        origin: koharu_scene::Origin::User,
                        name: "Cleanup".to_owned(),
                        kind: koharu_scene::RasterLayerKind::Cleanup,
                    },
                )?;
                edit.set_asset(
                    cleanup_layer,
                    &AssetRole::new("source")?,
                    AssetInput::new(
                        cleanup.clone(),
                        "image/png",
                        AssetMetadata {
                            width: Some(849),
                            height: Some(1200),
                            attributes: BTreeMap::new(),
                        },
                    ),
                )?;
                let source = edit.add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(
                        source_bounds.x,
                        source_bounds.y,
                        source_bounds.width,
                        source_bounds.height,
                    ),
                    Some("source".to_owned()),
                )?;
                edit.set(
                    source,
                    &DetectionAnalysis {
                        origin: koharu_scene::Origin::Generated(generation.clone()),
                        labels: vec![DetectionLabel {
                            kind: TextRegion::kind(),
                            confidence: 0.98,
                        }],
                    },
                )?;
                edit.set(
                    source,
                    &OcrAnalysis {
                        origin: koharu_scene::Origin::Generated(generation.clone()),
                        direction: TextDirection::Vertical,
                        confidence: Some(0.98),
                        line_boundaries: Vec::new(),
                    },
                )?;
                let target = edit.add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(
                        selected.bounds.x,
                        selected.bounds.y,
                        selected.bounds.width,
                        selected.bounds.height,
                    ),
                    Some("source-raster-adjacent-free-dialogue-anchor".to_owned()),
                )?;
                edit.set(
                    target,
                    &DetectionAnalysis {
                        origin: koharu_scene::Origin::Generated(generation.clone()),
                        labels: vec![DetectionLabel {
                            kind: RegionKind::new(ADJACENT_FREE_DIALOGUE_DETECTION_KIND)?,
                            confidence: selected.association.confidence as f32,
                        }],
                    },
                )?;
                let content = edit.add_text_content(page, At::End)?;
                let layer = edit.add_text_layer(
                    page,
                    At::End,
                    content,
                    &TextLayout {
                        origin: koharu_scene::Origin::User,
                        kind: TextLayoutKind::Paragraph,
                    },
                )?;
                edit.relate::<koharu_scene::RecognizedFrom>(content, source)?;
                edit.relate::<FitsTo>(layer, source)?;
                edit.set(
                    content,
                    &SourceText {
                        text: Authored::user("え～".to_owned()),
                        language: Some(LanguageTag::new("ja-JP")?),
                    },
                )?;
                edit.set(
                    content,
                    &Translation {
                        text: Authored::user("에~".to_owned()),
                        language: Some(LanguageTag::new("ko-KR")?),
                    },
                )?;
                edit.set(
                    content,
                    &TextRole {
                        origin: koharu_scene::Origin::User,
                        role: FREE_TEXT_ROLE.to_owned(),
                    },
                )?;
                edit.set(
                    layer,
                    &Typography {
                        origin: koharu_scene::Origin::User,
                        preferred_font: None,
                        font_weight: None,
                        font_style: None,
                        size: Some(17.425_472),
                        auto_fit: true,
                        color: Some([0, 0, 0, 255]),
                        stroke_color: None,
                        stroke_width: None,
                        alignment: None,
                        writing_mode: Some(WritingMode::Horizontal),
                        extensions: Default::default(),
                    },
                )?;
                edit.set(
                    layer,
                    &Geometry::rectangle(
                        source_bounds.x,
                        source_bounds.y,
                        source_bounds.width,
                        source_bounds.height,
                    ),
                )?;
                ids = Some((page, source, target, content, layer));
                Ok::<_, koharu_scene::Error>(())
            })()
            .unwrap();
            edit.finish().unwrap()
        };
        let snapshot = session.commit(setup).await.unwrap().snapshot;
        let (page, source, target, content, layer) = ids.unwrap();
        let decision = FreeDialogueAnchorDecision {
            schema_version: FREE_DIALOGUE_ANCHOR_SCHEMA_VERSION,
            decision: "source_raster_verified_adjacent_free_dialogue_anchor",
            evidence_revision: snapshot.revision(),
            decision_revision: snapshot.revision(),
            page_id: page,
            element_id: layer,
            content_id: content,
            source_region_id: source,
            target_region_id: target,
            source_ocr: SemanticText {
                text: "え～".to_owned(),
                language: Some("ja-JP".to_owned()),
            },
            target_translation: SemanticText {
                text: "에~".to_owned(),
                language: Some("ko-KR".to_owned()),
            },
            source_bounds,
            candidate_bounds: selected.bounds,
            distance_px: selected.distance_px,
            direction: selected.direction,
            pixel_analysis: selected.pixels.clone(),
            contrast_background_evidence: selected.contrast.clone(),
            room_evidence: selected.room.clone(),
            association: selected.association.clone(),
            writing_modes: assessment.writing_mode_gate.clone(),
            role_gate,
            deterministic_score: selected.score,
            deterministic_score_threshold: selected.score_threshold,
            association_confidence: selected.association.confidence,
            association_reason: selected.association.reason.clone(),
            detector_producer: RASTER_FREE_DIALOGUE_DETECTOR,
            detector_version: RASTER_FREE_DIALOGUE_DETECTOR_VERSION,
            detector_input: "original_source_pixels",
            rejected_candidates: assessment
                .candidates
                .iter()
                .filter(|candidate| !candidate.accepted)
                .cloned()
                .collect(),
        };
        host.free_dialogue_anchors
            .lock()
            .insert(layer, decision.clone());

        let renderer = Renderer::new().unwrap();
        let stale_source_frame = renderer.render(&snapshot, page).await.unwrap();
        let source_geometry_before = snapshot.component::<Geometry>(layer).unwrap().unwrap();
        let precommit = host.inspect_snapshot(snapshot.clone()).await.unwrap();
        let precommit_element = &precommit.project.pages[0].text_elements[0];
        assert_eq!(
            snapshot
                .relation_from::<FitsTo>(layer)
                .unwrap()
                .unwrap()
                .value()
                .target,
            source
        );
        assert_eq!(
            precommit_element.text_safe_region.as_ref().unwrap().id,
            source,
            "candidate evidence must not become the authoritative text-safe target before commit"
        );
        let precommit_acceptance = evaluate(&precommit);
        assert_eq!(
            precommit_acceptance.pages[0].elements[0]
                .measurements
                .target_layout_anchor
                .kind,
            crate::acceptance::LayoutAnchorKind::SourceRegion
        );
        assert_eq!(
            precommit_element.final_scene.layout_bounds,
            render_bounds(match stale_source_frame.layer(layer).unwrap().kind() {
                LayerKind::Text(metadata) => metadata.layout_bounds,
                LayerKind::Image(_) => panic!("expected translated text layer"),
            })
        );
        assert_eq!(
            snapshot.component::<Geometry>(layer).unwrap().unwrap(),
            source_geometry_before
        );

        assert_eq!(
            verified_free_dialogue_layout_target(&snapshot, layer, content, Some(&decision))
                .unwrap(),
            Some(target)
        );
        let fit = snapshot.relation_from::<FitsTo>(layer).unwrap().unwrap();
        assert!(
            stale_source_frame.layer(layer).unwrap().bounds().y
                > (candidate_bounds.y + candidate_bounds.height) as f32
        );
        let migration = ControlledTargetMigration {
            target,
            relation_to_remove: fit.id(),
            source_region: source,
            add_inside: false,
            relation: ControlledTargetRelation::FitsTo,
            layout_geometry: Some(Geometry::rectangle(
                candidate_bounds.x,
                candidate_bounds.y + 2.0,
                candidate_bounds.width,
                candidate_bounds.height - 2.0,
            )),
            layout_typography: Some(Typography {
                origin: koharu_scene::Origin::User,
                preferred_font: None,
                font_weight: None,
                font_style: None,
                size: Some(12.0),
                auto_fit: false,
                color: Some([0, 0, 0, 255]),
                stroke_color: None,
                stroke_width: None,
                alignment: None,
                writing_mode: Some(WritingMode::Horizontal),
                extensions: Default::default(),
            }),
        };
        let patch = snapshot
            .patch(|edit| apply_controlled_target_migration(edit, layer, migration))
            .unwrap();
        let candidate = snapshot.preview([&patch]).unwrap();

        assert_eq!(
            candidate
                .relation_from::<FitsTo>(layer)
                .unwrap()
                .unwrap()
                .value()
                .target,
            target
        );
        assert!(candidate.relation_from::<FlowsIn>(layer).unwrap().is_none());
        assert_eq!(
            element_geometry(candidate.component::<Geometry>(layer).unwrap().unwrap()).bounds,
            ElementBounds {
                x: candidate_bounds.x,
                y: candidate_bounds.y + 2.0,
                width: candidate_bounds.width,
                height: candidate_bounds.height - 2.0,
            }
        );
        assert_eq!(
            candidate
                .relation_from::<koharu_scene::RecognizedFrom>(content)
                .unwrap()
                .unwrap()
                .value()
                .target,
            source
        );
        assert!(
            candidate
                .component::<DetectionAnalysis>(target)
                .unwrap()
                .is_some()
        );

        let frame = renderer.render(&candidate, page).await.unwrap();
        let LayerKind::Text(metadata) = frame.layer(layer).unwrap().kind() else {
            panic!("expected translated text layer");
        };
        assert!(metadata.font_size >= 12.0);
        let cropped = frame.cropped(layer).unwrap().unwrap();
        let rasterizer = Rasterizer::new().unwrap();
        let raster = rasterizer
            .rasterize(
                &cropped.raster_frame().unwrap(),
                koharu_rasterizer::RasterOptions::default(),
            )
            .unwrap();
        let mut ink_pixels = 0;
        let mut ink_bounds = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for (index, pixel) in raster.image.pixels().enumerate() {
            if pixel[3] == 0 {
                continue;
            }
            ink_pixels += 1;
            let x = f64::from(raster.left) + (index as u32 % raster.image.width()) as f64 + 0.5;
            let y = f64::from(raster.top) + (index as u32 / raster.image.width()) as f64 + 0.5;
            ink_bounds[0] = ink_bounds[0].min(x);
            ink_bounds[1] = ink_bounds[1].min(y);
            ink_bounds[2] = ink_bounds[2].max(x);
            ink_bounds[3] = ink_bounds[3].max(y);
        }
        assert!(
            ink_pixels > 0,
            "preview must contain real rasterized glyph ink"
        );
        assert!(
            ink_bounds[0] >= candidate_bounds.x + 4.0
                && ink_bounds[2] <= candidate_bounds.x + candidate_bounds.width - 4.0
                && ink_bounds[1] >= candidate_bounds.y + 4.0
                && ink_bounds[3] <= candidate_bounds.y + candidate_bounds.height - 4.0,
            "preview raster ink {ink_bounds:?} escaped the verified candidate's 4px interior {candidate_bounds:?}"
        );

        let full_page = rasterizer
            .rasterize(
                &frame.raster_frame().unwrap(),
                koharu_rasterizer::RasterOptions::default(),
            )
            .unwrap()
            .image;
        assert!(source_pixels.iter().all(|&(x, y)| {
            let pixel = full_page.get_pixel(x, y);
            pixel[0] > 240 && pixel[1] > 240 && pixel[2] > 240
        }));
        let target_ink_pixels = (candidate_bounds.y + 4.0).ceil() as u32
            ..(candidate_bounds.y + candidate_bounds.height - 4.0).ceil() as u32;
        let target_ink_pixels = target_ink_pixels
            .flat_map(|y| {
                ((candidate_bounds.x + 4.0).ceil() as u32
                    ..(candidate_bounds.x + candidate_bounds.width - 4.0).ceil() as u32)
                    .map(move |x| (x, y))
            })
            .filter(|&(x, y)| {
                let pixel = full_page.get_pixel(x, y);
                u16::from(pixel[0]) + u16::from(pixel[1]) + u16::from(pixel[2]) < 3 * 128
            })
            .count();
        assert!(target_ink_pixels > 0);

        let mut inspected_page = sfx_classification_page(FREE_TEXT_ROLE, "え～");
        inspected_page.id = page;
        inspected_page.width = 849.0;
        inspected_page.height = 1200.0;
        let inspected = &mut inspected_page.text_elements[0];
        inspected.id = layer;
        inspected.content_id = content;
        inspected.source_region_id = Some(source);
        inspected.source.as_mut().unwrap().language = Some("ja-JP".to_owned());
        inspected.translation = Some(SemanticText {
            text: "에~".to_owned(),
            language: Some("ko-KR".to_owned()),
        });
        inspected.source_writing_mode = Some(WritingMode::Vertical);
        inspected.source_geometry = Some(element_geometry(
            candidate.component::<Geometry>(source).unwrap().unwrap(),
        ));
        inspected.text_safe_region = Some(TextSafeRegion {
            id: target,
            kind: TextRegion::KIND.to_owned(),
            geometry: element_geometry(candidate.component::<Geometry>(target).unwrap().unwrap()),
            association: Some(TargetRegionAssociation {
                layout_relation: TargetLayoutRelation::FitsTo,
                source_inside_target_relation: false,
            }),
        });
        inspected.verified_free_dialogue_anchor = Some(decision);
        inspected.typography = candidate.component::<Typography>(layer).unwrap();
        inspected.authored_layout_geometry = candidate
            .component::<Geometry>(layer)
            .unwrap()
            .map(element_geometry);
        inspected.final_scene = FinalSceneElement {
            eligible: true,
            visible: true,
            opacity: 1.0,
            geometry_visible: true,
            glyph_bounds: render_bounds(metadata.rendered_bounds),
            layout_bounds: render_bounds(metadata.layout_bounds),
            font_size_px: Some(f64::from(metadata.font_size)),
            line_count: Some(metadata.line_count),
            rendered_lines: metadata.rendered_lines.clone(),
            diagnostics: Vec::new(),
            glyph_ink: Some(GlyphInkMask {
                left: raster.left,
                top: raster.top,
                width: raster.image.width(),
                height: raster.image.height(),
                alpha: raster.image.pixels().map(|pixel| pixel[3]).collect(),
            }),
        };
        let acceptance = evaluate(&ProjectInspection {
            project: ProjectState {
                kind: "isolated_disposable",
                revision: candidate.revision(),
                pages: vec![inspected_page],
            },
            configuration: HarnessConfiguration {
                source_language: "ja-JP".to_owned(),
                target_language: "ko-KR".to_owned(),
                ocr_model: "test".to_owned(),
                required_output_directory: String::new(),
                required_export_format: "png".to_owned(),
                quality_thresholds: QualityThresholds::default(),
                review_bundle_directory: String::new(),
                external_visual_judge_configured: false,
            },
            repair_history: RepairHistory {
                completed_review_attempts: 0,
                attempted_actions: Vec::new(),
                active_deterministic_plan: None,
                stop_diagnostic: None,
                correction_actions: Vec::new(),
            },
        });
        assert!(
            acceptance.accepted,
            "full composited adjacent preview failed acceptance: {:?}",
            acceptance.rejection_reasons
        );

        let committed = session.commit(patch).await.unwrap().snapshot;
        let committed_inspection = host.inspect_snapshot(committed.clone()).await.unwrap();
        let committed_element = &committed_inspection.project.pages[0].text_elements[0];
        assert_eq!(
            committed
                .relation_from::<FitsTo>(layer)
                .unwrap()
                .unwrap()
                .value()
                .target,
            target
        );
        assert_eq!(
            committed_element.text_safe_region.as_ref().unwrap().id,
            target
        );
        let committed_acceptance = evaluate(&committed_inspection);
        assert_eq!(
            committed_acceptance.pages[0].elements[0]
                .measurements
                .target_layout_anchor
                .kind,
            crate::acceptance::LayoutAnchorKind::AdjacentFreeDialogueTextSafeAnchor
        );
        assert_eq!(
            element_geometry(committed.component::<Geometry>(layer).unwrap().unwrap()).bounds,
            ElementBounds {
                x: candidate_bounds.x,
                y: candidate_bounds.y + 2.0,
                width: candidate_bounds.width,
                height: candidate_bounds.height - 2.0,
            }
        );
    }

    fn compact_metrics(
        font_size: f64,
        accepted: bool,
        region_id: EntityId,
    ) -> TextLayoutPreviewMetrics {
        let mut metrics = layout_metrics(4, false, 0.0);
        metrics.deterministic_element_accepted = accepted;
        metrics.deterministic_rejection_codes = if accepted {
            Vec::new()
        } else {
            vec!["rendered_font_size_below_minimum".to_owned()]
        };
        metrics.rendered_font_size_px = Some(font_size);
        metrics.rendered_glyph_height_px = Some(11.5);
        metrics.target_layout_anchor = Some(crate::acceptance::LayoutAnchorMeasurement {
            kind: crate::acceptance::LayoutAnchorKind::TextSafeRegion,
            region_id: Some(region_id),
            region_kind: Some("dev.koharu.region.bubble".to_owned()),
            bounds: Some(ElementBounds {
                x: 4.0,
                y: 4.0,
                width: 92.0,
                height: 92.0,
            }),
            source_writing_mode: Some(WritingMode::Vertical),
            target_writing_mode: Some(WritingMode::Horizontal),
            association_confidence: Some(1.0),
            association_reason: "verified_japanese_vertical_to_korean_horizontal_text_safe_target"
                .to_owned(),
            coverage_policy:
                crate::acceptance::LayoutAnchorCoveragePolicy::ConfiguredLongDialogueFloor,
            expected_text_units: 22,
            target_anchor_area_px2: Some(8464.0),
            expected_ink_area_px2: 1320.0,
            measured_ink_bounds_area_px2: Some(846.4),
            rendered_area_coverage: Some(0.10),
            required_area_coverage: Some(0.08),
            rendered_overflow_px: Some(0.0),
        });
        metrics.text_safe_clearance = Some(text_safe_measurement(region_id, 5.0));
        metrics
    }

    #[tokio::test]
    async fn valid_shorter_compact_preview_commit_reaches_minimum_and_preserves_group_membership() {
        let mut session = Session::memory().await.unwrap();
        let mut ids = None;
        let setup = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(PageDraft::new("page", 100.0, 100.0), At::End)?;
                let target = edit.add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(0.0, 0.0, 100.0, 100.0),
                    None,
                )?;
                let first_region = edit.add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(10.0, 10.0, 20.0, 70.0),
                    None,
                )?;
                let second_region = edit.add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(35.0, 10.0, 20.0, 70.0),
                    None,
                )?;
                let first_content = edit.add_text_content(page, At::End)?;
                let first_layer = edit.add_text_layer(
                    page,
                    At::End,
                    first_content,
                    &TextLayout {
                        origin: koharu_scene::Origin::User,
                        kind: TextLayoutKind::Paragraph,
                    },
                )?;
                let second_content = edit.add_text_content(page, At::End)?;
                let second_layer = edit.add_text_layer(
                    page,
                    At::End,
                    second_content,
                    &TextLayout {
                        origin: koharu_scene::Origin::User,
                        kind: TextLayoutKind::Paragraph,
                    },
                )?;
                let ja = LanguageTag::new("ja-JP")?;
                edit.set(
                    first_content,
                    &SourceText {
                        text: Authored::user("一つ目".to_owned()),
                        language: Some(ja.clone()),
                    },
                )?;
                edit.set(
                    second_content,
                    &SourceText {
                        text: Authored::user("二つ目".to_owned()),
                        language: Some(ja),
                    },
                )?;
                edit.set(
                    first_content,
                    &Translation {
                        text: Authored::user(
                            "내 옷은 냉방 기능이 있으니까 괜찮아! 그리고 기쁘기도 하고요!"
                                .to_owned(),
                        ),
                        language: Some(LanguageTag::new("ko-KR")?),
                    },
                )?;
                let members = vec![
                    koharu_scene::LogicalDialogueMember {
                        ordinal: 1,
                        element_id: first_layer,
                        content_id: first_content,
                        source_region_id: first_region,
                    },
                    koharu_scene::LogicalDialogueMember {
                        ordinal: 2,
                        element_id: second_layer,
                        content_id: second_content,
                        source_region_id: second_region,
                    },
                ];
                edit.set(
                    first_content,
                    &koharu_scene::LogicalDialogue {
                        origin: koharu_scene::Origin::User,
                        target_region_id: target,
                        members: members.clone(),
                    },
                )?;
                ids = Some((first_content, first_layer, target, members));
                Ok(())
            })
            .unwrap();
        session.commit(setup).await.unwrap();
        let (content, _primary, target, members) = ids.unwrap();
        let base_revision = session.snapshot().revision();
        let before_group = session
            .snapshot()
            .component::<koharu_scene::LogicalDialogue>(content)
            .unwrap()
            .unwrap();
        let shorter = "냉방복이라 괜찮아! 기쁘기도 하고요!";
        assert!(
            validate_compact_translation_length(
                "내 옷은 냉방 기능이 있으니까 괜찮아! 그리고 기쁘기도 하고요!",
                shorter,
            )
            .is_ok()
        );
        let patch = session
            .snapshot()
            .patch(|edit| {
                edit.set(
                    content,
                    &Translation {
                        text: Authored::user(shorter.to_owned()),
                        language: Some(LanguageTag::new("ko-KR")?),
                    },
                )
            })
            .unwrap();
        let before_metrics = compact_metrics(10.8867, false, target);
        let candidate_metrics = compact_metrics(12.25, true, target);
        validate_compact_translation_candidate(
            &before_metrics,
            &candidate_metrics,
            &QualityThresholds::default(),
        )
        .unwrap();
        let committed = commit_compact_translation_patch_if_safe(
            &mut session,
            patch,
            base_revision,
            &before_metrics,
            &candidate_metrics,
            &QualityThresholds::default(),
        )
        .await
        .unwrap();

        assert_eq!(committed, Revision::new(base_revision.get() + 1));
        assert!(candidate_metrics.rendered_font_size_px.unwrap() >= 12.0);
        let after_group = session
            .snapshot()
            .component::<koharu_scene::LogicalDialogue>(content)
            .unwrap()
            .unwrap();
        assert_eq!(after_group, before_group);
        assert_eq!(after_group.members, members);
    }

    #[tokio::test]
    async fn stale_compact_preview_is_rejected_without_mutation() {
        let mut session = Session::memory().await.unwrap();
        let setup = session
            .snapshot()
            .patch(|edit| {
                edit.add_page(PageDraft::new("page", 100.0, 100.0), At::End)?;
                Ok(())
            })
            .unwrap();
        session.commit(setup).await.unwrap();
        let preview_revision = session.snapshot().revision();
        let stale_patch = session
            .snapshot()
            .patch(|edit| {
                edit.add_page(PageDraft::new("stale candidate", 20.0, 20.0), At::End)?;
                Ok(())
            })
            .unwrap();
        let intervening = session
            .snapshot()
            .patch(|edit| {
                edit.add_page(PageDraft::new("intervening", 30.0, 30.0), At::End)?;
                Ok(())
            })
            .unwrap();
        session.commit(intervening).await.unwrap();
        let revision_before_rejection = session.snapshot().revision();
        let region = EntityId::new();

        assert!(
            commit_compact_translation_patch_if_safe(
                &mut session,
                stale_patch,
                preview_revision,
                &compact_metrics(10.8, false, region),
                &compact_metrics(12.2, true, region),
                &QualityThresholds::default(),
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("stale")
        );
        assert_eq!(session.snapshot().revision(), revision_before_rejection);
        assert_eq!(session.snapshot().pages().len(), 2);
    }

    #[test]
    fn partial_typography_repair_preserves_unspecified_intent() {
        let existing = Typography {
            origin: koharu_scene::Origin::User,
            preferred_font: Some("Noto Sans KR".to_owned()),
            font_weight: Some(700),
            font_style: Some(FontStyle::Italic),
            size: Some(24.0),
            auto_fit: true,
            color: Some([1, 2, 3, 255]),
            stroke_color: Some([255, 255, 255, 255]),
            stroke_width: Some(2.0),
            alignment: Some(TextAlignment::Center),
            writing_mode: Some(WritingMode::Horizontal),
            extensions: Default::default(),
        };
        let repair = RepairTypography {
            preferred_font: None,
            font_weight: None,
            font_style: None,
            size: Some(20.0),
            auto_fit: None,
            color: None,
            stroke_color: None,
            stroke_width: None,
            alignment: None,
            writing_mode: None,
        };
        let revised = repair.to_scene(Some(&existing));
        assert_eq!(revised.size, Some(20.0));
        assert_eq!(revised.preferred_font, existing.preferred_font);
        assert_eq!(revised.font_weight, existing.font_weight);
        assert_eq!(revised.auto_fit, existing.auto_fit);
        assert_eq!(revised.alignment, existing.alignment);
    }

    #[test]
    fn typography_review_rejection_requires_a_layout_or_typography_repair() {
        let revision = koharu_scene::Revision::new(7);
        let mut review = review(VisualReviewStatus::Rejected, true, revision);
        review.decision = Some(crate::review::VisualReviewDecision {
            accepted: false,
            summary: "text crosses the balloon wall".to_owned(),
            issues: vec!["lower dialogue touches the left outline".to_owned()],
            judgments: crate::review::RequiredJudgments {
                source_text_accurate: true,
                translation_meaning_accurate: true,
                target_language_natural: true,
                reading_order_preserved: true,
                content_complete_without_duplicates: true,
                typography_layout_acceptable: false,
                skipped_items_are_difficult_sfx_and_no_required_content_skipped: true,
            },
        });
        assert!(review_requires_layout_repair(
            &review,
            None,
            EntityId::new()
        ));
    }

    #[test]
    fn repair_retries_are_bounded_by_exact_action_identity() {
        let first_element = EntityId::new();
        let second_element = EntityId::new();
        let first = RepairActionIdentity {
            operation: "controlled_source_raster_verified_free_dialogue_target_layout",
            element: first_element,
        };
        let same_operation_different_target = RepairActionIdentity {
            operation: first.operation,
            element: second_element,
        };
        let fallback = RepairActionIdentity {
            operation: "controlled_source_bound_vertical_interjection_fallback",
            element: first_element,
        };
        let mut attempted = BTreeSet::new();

        assert!(record_repair_action_attempt(&mut attempted, first.clone()));
        assert!(!record_repair_action_attempt(&mut attempted, first));
        assert!(record_repair_action_attempt(
            &mut attempted,
            same_operation_different_target
        ));
        assert!(record_repair_action_attempt(&mut attempted, fallback));
    }

    #[test]
    fn deterministic_page_clipping_plan_rejects_wrong_element_and_semantic_repair() {
        let wrong_element = EntityId::new();
        let failing_middle_element = EntityId::new();
        let revision = koharu_scene::Revision::new(11);
        let plan = deterministic_repair_plan(
            &[crate::acceptance::AcceptanceRejection {
                code: AcceptanceRejectionCode::RenderedTextOutsidePage,
                page_id: EntityId::new(),
                element_id: Some(failing_middle_element),
                related_element_id: None,
                expected: json!({ "maximum_px": 0.5 }),
                actual: json!({ "value_px": 3.27 }),
            }],
            revision,
            None,
        );
        let output = serde_json::to_value(&plan).unwrap();
        assert_eq!(
            output["schema_version"],
            crate::repair::REPAIR_PLAN_SCHEMA_VERSION
        );
        assert_eq!(output["unresolved_failure_count"], 1);
        assert_eq!(
            output["blocking_failures"][0]["element_id"],
            failing_middle_element.to_string()
        );
        assert_eq!(
            output["blocking_failures"][0]["code"],
            "rendered_text_outside_page"
        );
        assert_eq!(output["blocking_failures"][0]["actual"]["value_px"], 3.27);
        assert_eq!(
            output["blocking_failures"][0]["required_evidence_revision"],
            11
        );
        assert_eq!(
            output["blocking_failures"][0]["next_action"]["tool"],
            "preview_text_layout"
        );

        let wrong = planned_repair_failure(&plan, wrong_element)
            .unwrap_err()
            .to_string();
        assert!(wrong.contains(&failing_middle_element.to_string()));

        let failure = planned_repair_failure(&plan, failing_middle_element)
            .unwrap()
            .unwrap();
        assert_eq!(failure.required_evidence_revision, revision);
        assert_eq!(
            failure.allowed_repair_fields,
            [RepairField::Typography, RepairField::Layout]
        );
        let next_action = failure.next_action.as_ref().unwrap();
        assert_eq!(next_action.tool, "preview_text_layout");
        assert_eq!(next_action.operation, "controlled_text_layout");
        assert_eq!(next_action.element, failing_middle_element);
        assert!(validate_planned_repair_fields(failure, &["translation_text"]).is_err());
        assert!(validate_planned_repair_fields(failure, &["layout_geometry"]).is_ok());
    }

    #[test]
    fn deterministic_plan_stops_a_no_progress_repair_review_loop() {
        let element = EntityId::new();
        let rejection = |overflow, revision| {
            deterministic_repair_plan(
                &[crate::acceptance::AcceptanceRejection {
                    code: AcceptanceRejectionCode::RenderedTextOutsidePage,
                    page_id: EntityId::new(),
                    element_id: Some(element),
                    related_element_id: None,
                    expected: json!({ "maximum_px": 0.5 }),
                    actual: json!({ "value_px": overflow }),
                }],
                revision,
                None,
            )
        };
        let previous = rejection(3.27, koharu_scene::Revision::new(11));
        let current = rejection(3.51, koharu_scene::Revision::new(12));

        let diagnostic = repair_stop_diagnostic(&previous, &current).unwrap();
        assert_eq!(diagnostic.previous_unresolved_failure_count, 1);
        assert_eq!(diagnostic.current_unresolved_failure_count, 1);
        assert_eq!(
            diagnostic.reviewed_revision,
            koharu_scene::Revision::new(12)
        );

        let resolved = deterministic_repair_plan(&[], koharu_scene::Revision::new(12), None);
        assert!(repair_stop_diagnostic(&previous, &resolved).is_none());

        let mut distinct = current;
        distinct.blocking_failures[0]
            .next_action
            .as_mut()
            .unwrap()
            .operation = "controlled_source_bound_vertical_interjection_fallback";
        assert!(repair_stop_diagnostic(&previous, &distinct).is_none());
    }

    #[test]
    fn source_self_fits_to_required_ui_clearance_is_not_a_terminal_repair_failure() {
        let revision = Revision::new(7);
        let mut page = sfx_classification_page("dev.koharu.text.ui", "故障中");
        let element = &mut page.text_elements[0];
        let element_id = element.id;
        let source_anchor_id = element.source_region_id.unwrap();
        let bounds = ElementBounds {
            x: 181.573_242_187_5,
            y: 501.562_5,
            width: 68.815_429_687_5,
            height: 18.75,
        };
        let geometry = ElementGeometry {
            points: vec![
                ElementPoint {
                    x: bounds.x,
                    y: bounds.y,
                },
                ElementPoint {
                    x: bounds.x + bounds.width,
                    y: bounds.y,
                },
                ElementPoint {
                    x: bounds.x + bounds.width,
                    y: bounds.y + bounds.height,
                },
                ElementPoint {
                    x: bounds.x,
                    y: bounds.y + bounds.height,
                },
            ],
            bounds,
        };
        element.source_geometry = Some(geometry.clone());
        element.text_safe_region = Some(TextSafeRegion {
            id: source_anchor_id,
            kind: "dev.koharu.region.text".to_owned(),
            geometry,
            association: Some(TargetRegionAssociation {
                layout_relation: TargetLayoutRelation::FitsTo,
                source_inside_target_relation: false,
            }),
        });
        element.source_writing_mode = Some(WritingMode::Horizontal);
        element.translation = Some(SemanticText {
            text: "고장 중".to_owned(),
            language: Some("ko-KR".to_owned()),
        });
        element.final_scene = FinalSceneElement {
            eligible: true,
            visible: true,
            opacity: 1.0,
            geometry_visible: true,
            glyph_bounds: Some(bounds),
            layout_bounds: Some(bounds),
            font_size_px: Some(18.572_513_580_322_266),
            line_count: Some(1),
            rendered_lines: vec!["고장 중".to_owned()],
            diagnostics: Vec::new(),
            glyph_ink: None,
        };
        let inspection = ProjectInspection {
            project: ProjectState {
                kind: "isolated_disposable",
                revision,
                pages: vec![page],
            },
            configuration: HarnessConfiguration {
                source_language: "ja-JP".to_owned(),
                target_language: "ko-KR".to_owned(),
                ocr_model: "test".to_owned(),
                required_output_directory: String::new(),
                required_export_format: "png".to_owned(),
                quality_thresholds: QualityThresholds::default(),
                review_bundle_directory: String::new(),
                external_visual_judge_configured: false,
            },
            repair_history: RepairHistory {
                completed_review_attempts: 0,
                attempted_actions: Vec::new(),
                active_deterministic_plan: None,
                stop_diagnostic: None,
                correction_actions: Vec::new(),
            },
        };
        assert!(!is_actual_container_bound(
            &inspection.project.pages[0].text_elements[0]
        ));
        let rejection = AcceptanceRejection {
            code: AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior,
            page_id: inspection.project.pages[0].id,
            element_id: Some(element_id),
            related_element_id: None,
            expected: json!({
                "minimum_clearance_px": 4.0,
                "target_layout_anchor": {
                    "kind": "source_region",
                    "target_writing_mode": "Horizontal"
                }
            }),
            actual: json!({
                "region_id": source_anchor_id,
                "minimum_clearance_px": 0.105_393_218_813_452_43
            }),
        };

        let plan = deterministic_repair_plan(&[rejection], revision, Some(&inspection));
        assert!(plan.terminal_diagnostic.is_none());
        assert!(plan.blocking_failures.is_empty());
        assert_eq!(plan.unresolved_failure_count, 0);
        assert!(infeasible_repair_stop_diagnostic(&plan).is_none());
    }

    #[test]
    fn deterministic_semantic_failure_allows_only_its_semantic_field() {
        let element = EntityId::new();
        let plan = deterministic_repair_plan(
            &[crate::acceptance::AcceptanceRejection {
                code: AcceptanceRejectionCode::SourceTextEmpty,
                page_id: EntityId::new(),
                element_id: Some(element),
                related_element_id: None,
                expected: json!("nonempty source text"),
                actual: json!(""),
            }],
            koharu_scene::Revision::new(5),
            None,
        );
        let failure = planned_repair_failure(&plan, element).unwrap().unwrap();

        assert!(validate_planned_repair_fields(failure, &["source_text"]).is_ok());
        assert!(validate_planned_repair_fields(failure, &["translation_text"]).is_err());
        assert!(validate_planned_repair_fields(failure, &["typography"]).is_err());

        let allowed = PageTranslationEdit {
            element: element.to_string(),
            source: Some(LanguageTextEdit {
                text: "corrected source".to_owned(),
                language: "en-US".to_owned(),
            }),
            translation: None,
        };
        let wrong_field = PageTranslationEdit {
            element: element.to_string(),
            source: None,
            translation: Some(LanguageTextEdit {
                text: "수정".to_owned(),
                language: "ko".to_owned(),
            }),
        };
        let wrong_element = EntityId::new();
        let wrong_target = PageTranslationEdit {
            element: wrong_element.to_string(),
            ..allowed.clone()
        };
        assert!(validate_page_semantic_plan(Some(failure), &[(element, &allowed)]).is_ok());
        assert!(validate_page_semantic_plan(Some(failure), &[(element, &wrong_field)]).is_err());
        assert!(
            validate_page_semantic_plan(Some(failure), &[(wrong_element, &wrong_target)]).is_err()
        );
    }

    #[test]
    fn pending_agent_review_allows_current_project_inspection_evidence() {
        let revision = koharu_scene::Revision::new(7);
        let page = EntityId::new();
        let observation = RevisionEvidence::ProjectInspection { revision };
        let review = review(VisualReviewStatus::PendingAgentReview, true, revision);

        assert_eq!(
            repair_evidence(&review, Some(&observation), revision, page).unwrap(),
            observation
        );
        assert!(!review.accepted());
    }

    #[test]
    fn pending_agent_review_requires_fresh_relevant_visual_evidence() {
        let current = koharu_scene::Revision::new(8);
        let stale = RevisionEvidence::ProjectInspection {
            revision: koharu_scene::Revision::new(7),
        };
        let page = EntityId::new();
        let other_page = EntityId::new();
        let wrong_page = RevisionEvidence::RenderedPageInspection {
            revision: current,
            page_id: other_page,
        };
        let review = review(VisualReviewStatus::PendingAgentReview, true, current);

        assert!(repair_evidence(&review, None, current, page).is_err());
        assert!(repair_evidence(&review, Some(&stale), current, page).is_err());
        assert!(repair_evidence(&review, Some(&wrong_page), current, page).is_err());
        assert_eq!(
            repair_evidence(
                &review,
                Some(&RevisionEvidence::RenderedPageInspection {
                    revision: current,
                    page_id: page,
                }),
                current,
                page,
            )
            .unwrap()
            .revision(),
            current
        );
    }

    #[test]
    fn accepted_external_review_cannot_be_used_as_repair_evidence() {
        let revision = koharu_scene::Revision::new(7);
        let observation = RevisionEvidence::ProjectInspection { revision };
        let review = review(VisualReviewStatus::Accepted, true, revision);

        assert!(repair_evidence(&review, Some(&observation), revision, EntityId::new()).is_err());
    }

    #[test]
    fn correction_trace_records_evidence_revision_rationale_and_original_ocr() {
        let revision = koharu_scene::Revision::new(7);
        let original = RepairText {
            text: "OCR before repair".to_owned(),
            language: Some("en-US".to_owned()),
            origin: koharu_scene::Origin::User,
        };
        let state = RepairElementState {
            source: Some(original.clone()),
            translation: None,
            typography: None,
            layout_kind: TextLayoutKind::Paragraph,
            authored_layout_geometry: None,
        };
        let record = CorrectionRecord {
            schema_version: CORRECTION_SCHEMA_VERSION,
            sequence: 1,
            review_attempt: 1,
            revision_before: revision,
            revision_after: koharu_scene::Revision::new(8),
            element_id: EntityId::new(),
            content_id: EntityId::new(),
            original_ocr: Some(original),
            evidence: RevisionEvidence::ProjectInspection { revision },
            changed_fields: vec!["source_text"],
            before: state.clone(),
            after: state,
            reason: "the inspected source glyph is a digit".to_owned(),
            agent_action: AgentAction {
                actor: "codex_agent",
                configured_model: None,
                tool: "revise_element",
                tool_call_id: "call-test".to_owned(),
            },
        };

        let traced = serde_json::to_value(record).unwrap();
        assert_eq!(traced["schema_version"], CORRECTION_SCHEMA_VERSION);
        assert_eq!(traced["evidence"]["kind"], "project_inspection");
        assert_eq!(traced["evidence"]["revision"], 7);
        assert_eq!(traced["reason"], "the inspected source glyph is a digit");
        assert_eq!(traced["original_ocr"]["text"], "OCR before repair");
    }

    #[test]
    fn later_corrections_keep_the_first_ocr_value() {
        let original = RepairText {
            text: "OCR before repair".to_owned(),
            language: Some("en-US".to_owned()),
            origin: koharu_scene::Origin::User,
        };
        let current = RepairText {
            text: "intermediate correction".to_owned(),
            language: Some("en-US".to_owned()),
            origin: koharu_scene::Origin::User,
        };
        assert_eq!(
            preserve_original_ocr(Some(original.clone()), Some(current)),
            Some(original)
        );
    }
}
