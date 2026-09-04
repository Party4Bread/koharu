use anyhow::{Result, bail};
use clap::Args;
use koharu_scene::{EntityId, RegionSpec, TextLayoutKind, Typography, WritingMode};
use serde::Serialize;
use serde_json::{Value, json};
use unicode_segmentation::UnicodeSegmentation as _;

use crate::free_dialogue::{
    FreeDialogueAnchorAssessment, FreeDialogueAnchorDecision,
    validate_free_dialogue_anchor_decision,
};
use crate::placement::is_actual_container_bound;
use crate::repair::RepairHistory;
use crate::sfx::{
    DECORATIVE_SFX_DECISION_SCHEMA_VERSION, DECORATIVE_SFX_ROLE, DecorativeSfxDecision,
    DecorativeSfxDisposition, FREE_TEXT_ROLE, SKIPPED_DIFFICULT_SFX_ROLE,
};
use crate::ui_panel::{
    DetectedPanelCandidate, UI_TEXT_ROLE, UiPanelAnchorDecision, raster_candidate_supports_ui_role,
};

pub(crate) const ACCEPTANCE_SCHEMA_VERSION: u32 = 13;
const MAX_RECORDED_TEXT_SAFE_VIOLATIONS: usize = 16;
pub(crate) const PIXEL_HALF_DIAGONAL: f64 = std::f64::consts::SQRT_2 * 0.5;
const RELIABLE_TARGET_REGION_CONTAINMENT: f64 = 0.90;
const MIN_VERIFIED_TARGET_AREA_COVERAGE: f64 = 0.01;
const EXPECTED_VISIBLE_UNIT_AREA_FACTOR: f64 = 0.50;
const MAX_REASONABLE_RENDERED_LINE_COUNT: usize = 12;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProjectInspection {
    pub project: ProjectState,
    pub configuration: HarnessConfiguration,
    pub repair_history: RepairHistory,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProjectState {
    pub kind: &'static str,
    pub revision: koharu_scene::Revision,
    pub pages: Vec<PageInspection>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct HarnessConfiguration {
    pub source_language: String,
    pub target_language: String,
    pub ocr_model: String,
    pub required_output_directory: String,
    pub required_export_format: String,
    pub quality_thresholds: QualityThresholds,
    pub review_bundle_directory: String,
    pub external_visual_judge_configured: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PageInspection {
    pub id: EntityId,
    pub label: String,
    pub width: f64,
    pub height: f64,
    pub text_elements: Vec<TextElementInspection>,
    pub detected_panel_candidates: Vec<DetectedPanelCandidate>,
    pub logical_dialogue_groups: Vec<LogicalDialogueGroupInspection>,
    pub render_error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LogicalDialogueGroupInspection {
    pub group_id: EntityId,
    pub primary_render_element_id: EntityId,
    pub target_region_id: EntityId,
    pub logical_source_text: String,
    pub members: Vec<LogicalDialogueMemberInspection>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LogicalDialogueMemberInspection {
    pub ordinal: u32,
    pub element_id: EntityId,
    pub content_id: EntityId,
    pub source_region_id: EntityId,
    pub source_text: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LogicalDialogueMembershipInspection {
    pub group_id: EntityId,
    pub primary_render_element_id: EntityId,
    pub target_region_id: EntityId,
    pub member_ordinal: u32,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TextElementInspection {
    pub id: EntityId,
    pub content_id: EntityId,
    pub source_region_id: Option<EntityId>,
    pub source_region_kind: Option<String>,
    pub detected: bool,
    pub required: bool,
    pub text_role: Option<String>,
    pub decorative_sfx: Option<DecorativeSfxDecision>,
    pub logical_dialogue_memberships: Vec<LogicalDialogueMembershipInspection>,
    pub source: Option<SemanticText>,
    pub translation: Option<SemanticText>,
    pub source_writing_mode: Option<WritingMode>,
    pub visibility: ElementVisibility,
    pub source_geometry: Option<ElementGeometry>,
    pub text_safe_region: Option<TextSafeRegion>,
    pub verified_ui_panel_anchor: Option<UiPanelAnchorDecision>,
    pub verified_free_dialogue_anchor: Option<FreeDialogueAnchorDecision>,
    pub free_dialogue_anchor_assessment: Option<FreeDialogueAnchorAssessment>,
    pub typography: Option<Typography>,
    pub layout_kind: TextLayoutKind,
    pub authored_layout_geometry: Option<ElementGeometry>,
    pub final_scene: FinalSceneElement,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TextSafeRegion {
    pub id: EntityId,
    pub kind: String,
    pub geometry: ElementGeometry,
    pub association: Option<TargetRegionAssociation>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TargetLayoutRelation {
    FitsTo,
    FlowsIn,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct TargetRegionAssociation {
    pub layout_relation: TargetLayoutRelation,
    pub source_inside_target_relation: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SemanticText {
    pub text: String,
    pub language: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct ElementVisibility {
    pub local_visible: bool,
    pub local_opacity: f32,
    pub effective_visible: bool,
    pub effective_opacity: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct ElementGeometry {
    pub points: Vec<ElementPoint>,
    pub bounds: ElementBounds,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) struct ElementPoint {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct ElementBounds {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FinalSceneElement {
    pub eligible: bool,
    pub visible: bool,
    pub opacity: f32,
    pub geometry_visible: bool,
    pub glyph_bounds: Option<ElementBounds>,
    pub layout_bounds: Option<ElementBounds>,
    pub font_size_px: Option<f64>,
    pub line_count: Option<usize>,
    pub rendered_lines: Vec<String>,
    pub diagnostics: Vec<String>,
    #[serde(skip)]
    pub glyph_ink: Option<GlyphInkMask>,
}

#[derive(Clone, Debug)]
pub(crate) struct GlyphInkMask {
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
    pub alpha: Vec<u8>,
}

impl Default for FinalSceneElement {
    fn default() -> Self {
        Self {
            eligible: false,
            visible: false,
            opacity: 0.0,
            geometry_visible: false,
            glyph_bounds: None,
            layout_bounds: None,
            font_size_px: None,
            line_count: None,
            rendered_lines: Vec::new(),
            diagnostics: Vec::new(),
            glyph_ink: None,
        }
    }
}

#[derive(Args, Clone, Debug, Serialize)]
pub struct QualityThresholds {
    /// Minimum fitted font size in page pixels.
    #[arg(long, default_value_t = 12.0)]
    pub min_rendered_font_size_px: f64,
    /// Minimum height of transformed rendered glyph bounds in page pixels.
    #[arg(long, default_value_t = 10.0)]
    pub min_rendered_glyph_height_px: f64,
    /// Minimum glyph intersection coverage for source anchors and verified target anchors whose
    /// grapheme-derived expected ink reaches this density. Short text in a verified dialogue or
    /// required UI-panel text-safe target uses the length-aware contract instead.
    #[arg(long, default_value_t = 0.08)]
    pub min_layout_anchor_area_coverage: f64,
    /// Maximum glyph overflow beyond its authoritative layout-anchor bounds in pixels.
    #[arg(long, default_value_t = 2.0)]
    pub max_layout_anchor_overflow_px: f64,
    /// Required glyph-ink clearance from the detected balloon, UI-panel, or text-region contour.
    #[arg(long, default_value_t = 4.0)]
    pub min_text_safe_padding_px: f64,
    /// Maximum glyph overflow beyond page bounds in pixels.
    #[arg(long, default_value_t = 0.5)]
    pub max_page_overflow_px: f64,
    /// Maximum intersection divided by the smaller rendered text-region area.
    #[arg(long, default_value_t = 0.20)]
    pub max_translated_region_overlap: f64,
    /// Maximum intersection divided by the smaller detected source-region area.
    #[arg(long, default_value_t = 0.75)]
    pub max_source_region_overlap: f64,
    /// Maximum source overlap for normalized-identical translations.
    #[arg(long, default_value_t = 0.60)]
    pub max_duplicate_source_region_overlap: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_rendered_font_size_px: 12.0,
            min_rendered_glyph_height_px: 10.0,
            min_layout_anchor_area_coverage: 0.08,
            max_layout_anchor_overflow_px: 2.0,
            min_text_safe_padding_px: 4.0,
            max_page_overflow_px: 0.5,
            max_translated_region_overlap: 0.20,
            max_source_region_overlap: 0.75,
            max_duplicate_source_region_overlap: 0.60,
        }
    }
}

impl QualityThresholds {
    pub(crate) fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("min_rendered_font_size_px", self.min_rendered_font_size_px),
            (
                "min_rendered_glyph_height_px",
                self.min_rendered_glyph_height_px,
            ),
            (
                "max_layout_anchor_overflow_px",
                self.max_layout_anchor_overflow_px,
            ),
            ("min_text_safe_padding_px", self.min_text_safe_padding_px),
            ("max_page_overflow_px", self.max_page_overflow_px),
        ] {
            if !value.is_finite() || value < 0.0 {
                bail!("{name} must be finite and nonnegative, got {value}");
            }
        }
        for (name, value) in [
            (
                "min_layout_anchor_area_coverage",
                self.min_layout_anchor_area_coverage,
            ),
            (
                "max_translated_region_overlap",
                self.max_translated_region_overlap,
            ),
            ("max_source_region_overlap", self.max_source_region_overlap),
            (
                "max_duplicate_source_region_overlap",
                self.max_duplicate_source_region_overlap,
            ),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                bail!("{name} must be between 0 and 1 inclusive, got {value}");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct AcceptanceRecord {
    pub schema_version: u32,
    pub accepted: bool,
    pub source_language: String,
    pub target_language: String,
    pub thresholds: QualityThresholds,
    pub pages: Vec<PageAcceptance>,
    pub rejection_reasons: Vec<AcceptanceRejection>,
    pub semantic_fidelity_requires_external_review: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AcceptanceCounts {
    pub detected_source_elements: usize,
    pub required_source_elements: usize,
    pub skipped_difficult_sfx: usize,
    pub source_text_present: usize,
    pub source_text_nonempty: usize,
    pub requested_source_language_elements: usize,
    pub translation_present: usize,
    pub translation_nonempty: usize,
    pub target_language_translations: usize,
    pub render_eligible_translations: usize,
    pub visible_translations: usize,
    pub accepted_elements: usize,
    pub pairwise_conflicts: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct PageAcceptance {
    pub page_id: EntityId,
    pub label: String,
    pub accepted: bool,
    pub counts: AcceptanceCounts,
    pub elements: Vec<ElementAcceptance>,
    pub pairs: Vec<PairAcceptance>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ElementAcceptance {
    pub element_id: EntityId,
    pub logical_group_id: Option<EntityId>,
    pub render_owner_element_id: EntityId,
    pub evaluated_render_owner: bool,
    pub accepted: bool,
    pub measurements: ElementMeasurements,
}

#[derive(Clone, Debug, Serialize)]
pub struct ElementMeasurements {
    pub requested_source_language: String,
    pub actual_source_language: Option<String>,
    pub source_language_matches: bool,
    pub source_region_bounds: Option<ElementBounds>,
    pub rendered_glyph_bounds: Option<ElementBounds>,
    pub layout_bounds: Option<ElementBounds>,
    pub rendered_font_size_px: Option<f64>,
    pub rendered_line_count: Option<usize>,
    pub maximum_reasonable_line_count: usize,
    pub rendered_line_count_reasonable: bool,
    pub rendered_glyph_height_px: Option<f64>,
    pub source_region_area_coverage: Option<f64>,
    pub source_region_overflow_px: Option<f64>,
    pub target_layout_anchor: LayoutAnchorMeasurement,
    pub(crate) free_dialogue_anchor_evidence: Option<FreeDialogueAnchorDecision>,
    pub text_safe_containment: Option<TextSafeContainment>,
    pub page_overflow_px: Option<f64>,
    pub finite_positive_layout: bool,
    pub renderer_diagnostics: Vec<String>,
    pub invalid_metrics: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutAnchorKind {
    SourceRegion,
    TextSafeRegion,
    UiPanelTextSafeInterior,
    AdjacentFreeDialogueTextSafeAnchor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutAnchorCoveragePolicy {
    ConfiguredSourceAnchorFloor,
    ConfiguredLongDialogueFloor,
    VerifiedTargetExpectedInk,
}

#[derive(Clone, Debug, Serialize)]
pub struct LayoutAnchorMeasurement {
    pub kind: LayoutAnchorKind,
    pub region_id: Option<EntityId>,
    pub region_kind: Option<String>,
    pub bounds: Option<ElementBounds>,
    pub source_writing_mode: Option<WritingMode>,
    pub target_writing_mode: Option<WritingMode>,
    pub association_confidence: Option<f64>,
    pub association_reason: String,
    pub coverage_policy: LayoutAnchorCoveragePolicy,
    pub expected_text_units: usize,
    pub target_anchor_area_px2: Option<f64>,
    pub expected_ink_area_px2: f64,
    pub measured_ink_bounds_area_px2: Option<f64>,
    pub rendered_area_coverage: Option<f64>,
    pub required_area_coverage: Option<f64>,
    pub rendered_overflow_px: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TextSafeContainment {
    pub region_id: EntityId,
    pub region_kind: String,
    pub region_bounds: ElementBounds,
    pub required_padding_px: f64,
    pub minimum_clearance_px: Option<f64>,
    pub glyph_ink_pixels: usize,
    pub violation_pixels: usize,
    pub violations: Vec<TextSafeViolation>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct TextSafeViolation {
    pub x: f64,
    pub y: f64,
    pub alpha: u8,
    pub clearance_px: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct PairAcceptance {
    pub first_element_id: EntityId,
    pub second_element_id: EntityId,
    pub accepted: bool,
    pub normalized_translation_equal: bool,
    pub rendered_region_overlap: Option<f64>,
    pub source_region_overlap: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AcceptanceRejection {
    pub code: AcceptanceRejectionCode,
    pub page_id: EntityId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub element_id: Option<EntityId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub related_element_id: Option<EntityId>,
    pub expected: Value,
    pub actual: Value,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceptanceRejectionCode {
    PageNoDetectedSourceText,
    InvalidDecorativeSfxDecision,
    SourceTextMissing,
    SourceTextEmpty,
    SourceLanguageMissing,
    SourceLanguageMismatch,
    TranslationMissing,
    TranslationEmpty,
    TranslationLanguageMissing,
    TranslationLanguageMismatch,
    LogicalDialogueOwnershipConflict,
    LogicalDialogueMemberMismatch,
    LogicalDialogueNonPrimaryTranslation,
    LogicalDialogueNonPrimaryVisible,
    TypesettingFailed,
    TranslationNotRenderEligible,
    TranslationHidden,
    TranslationZeroOpacity,
    TranslationGeometryNotVisible,
    SourceRegionGeometryMissing,
    TextSafeRegionMissing,
    RenderedGlyphInkMissing,
    UnsupportedLayout,
    RenderedFontSizeMissing,
    RenderedFontSizeBelowMinimum,
    RenderedLineCountMissing,
    RenderedLineCountUnreasonable,
    RenderedGlyphHeightBelowMinimum,
    SourceRegionCoverageBelowMinimum,
    RenderedTextOutsideSourceRegion,
    TargetAnchorCoverageBelowMinimum,
    RenderedTextOutsideTargetAnchor,
    RenderedTextOutsideTextSafeInterior,
    RenderedTextOutsidePage,
    DuplicateTranslationRegion,
    TranslatedRegionOverlap,
    SourceRegionOverlap,
}

pub(crate) fn evaluate(inspection: &ProjectInspection) -> AcceptanceRecord {
    let source_language = inspection.configuration.source_language.as_str();
    let target_language = inspection.configuration.target_language.as_str();
    let thresholds = &inspection.configuration.quality_thresholds;
    let mut pages = Vec::with_capacity(inspection.project.pages.len());
    let mut rejection_reasons = Vec::new();

    for page in &inspection.project.pages {
        let page_rejection_start = rejection_reasons.len();
        let mut counts = AcceptanceCounts::default();
        let mut elements = Vec::new();
        if let Some(error) = &page.render_error {
            rejection_reasons.push(AcceptanceRejection {
                code: AcceptanceRejectionCode::TypesettingFailed,
                page_id: page.id,
                element_id: None,
                related_element_id: None,
                expected: Value::String("page renders successfully".to_owned()),
                actual: Value::String(error.clone()),
            });
        }
        validate_logical_dialogue_groups(page, &mut rejection_reasons);

        for element in &page.text_elements {
            if element.detected {
                counts.detected_source_elements += 1;
            }
            let validated_translated_decorative_sfx = if let Some(decision) =
                &element.decorative_sfx
            {
                counts.skipped_difficult_sfx += usize::from(decision.is_skipped());
                validate_decorative_sfx_decision(page, element, decision, &mut rejection_reasons)
                    && decision.is_translated()
            } else {
                false
            };
            if !element.required {
                continue;
            }
            counts.required_source_elements += 1;
            let rejection_start = rejection_reasons.len();
            let logical_group =
                validate_element_logical_ownership(page, element, &mut rejection_reasons);
            let render_owner_element_id = logical_group.map_or(element.id, |membership| {
                membership.primary_render_element_id
            });
            let evaluated_render_owner = render_owner_element_id == element.id;

            match &element.source {
                None => reject(
                    &mut rejection_reasons,
                    AcceptanceRejectionCode::SourceTextMissing,
                    page.id,
                    element.id,
                    Value::String("source text component".to_owned()),
                    Value::Null,
                ),
                Some(source) => {
                    counts.source_text_present += 1;
                    if source.text.trim().is_empty() {
                        reject(
                            &mut rejection_reasons,
                            AcceptanceRejectionCode::SourceTextEmpty,
                            page.id,
                            element.id,
                            Value::String("nonempty source text".to_owned()),
                            Value::String(source.text.clone()),
                        );
                    } else {
                        counts.source_text_nonempty += 1;
                    }
                    match source.language.as_deref() {
                        None => reject(
                            &mut rejection_reasons,
                            AcceptanceRejectionCode::SourceLanguageMissing,
                            page.id,
                            element.id,
                            Value::String(source_language.to_owned()),
                            Value::Null,
                        ),
                        Some(language) if language != source_language => reject(
                            &mut rejection_reasons,
                            AcceptanceRejectionCode::SourceLanguageMismatch,
                            page.id,
                            element.id,
                            Value::String(source_language.to_owned()),
                            Value::String(language.to_owned()),
                        ),
                        Some(_) => counts.requested_source_language_elements += 1,
                    }
                }
            }

            if !evaluated_render_owner {
                if element.translation.is_some() {
                    reject(
                        &mut rejection_reasons,
                        AcceptanceRejectionCode::LogicalDialogueNonPrimaryTranslation,
                        page.id,
                        element.id,
                        Value::String(
                            "translation owned only by the logical group's primary render element"
                                .to_owned(),
                        ),
                        serde_json::to_value(&element.translation).unwrap_or(Value::Null),
                    );
                }
                if element.visibility.effective_visible || element.final_scene.visible {
                    reject(
                        &mut rejection_reasons,
                        AcceptanceRejectionCode::LogicalDialogueNonPrimaryVisible,
                        page.id,
                        element.id,
                        Value::String("non-primary logical dialogue member retained as non-rendering source evidence".to_owned()),
                        json!({
                            "effective_visible": element.visibility.effective_visible,
                            "final_scene_visible": element.final_scene.visible,
                        }),
                    );
                }
            } else {
                match &element.translation {
                    None => reject(
                        &mut rejection_reasons,
                        AcceptanceRejectionCode::TranslationMissing,
                        page.id,
                        element.id,
                        Value::String("translation component".to_owned()),
                        Value::Null,
                    ),
                    Some(translation) => {
                        counts.translation_present += 1;
                        if translation.text.trim().is_empty() {
                            reject(
                                &mut rejection_reasons,
                                AcceptanceRejectionCode::TranslationEmpty,
                                page.id,
                                element.id,
                                Value::String("nonempty translation".to_owned()),
                                Value::String(translation.text.clone()),
                            );
                        } else {
                            counts.translation_nonempty += 1;
                        }
                        match translation.language.as_deref() {
                            None => reject(
                                &mut rejection_reasons,
                                AcceptanceRejectionCode::TranslationLanguageMissing,
                                page.id,
                                element.id,
                                Value::String(target_language.to_owned()),
                                Value::Null,
                            ),
                            Some(language) if language != target_language => reject(
                                &mut rejection_reasons,
                                AcceptanceRejectionCode::TranslationLanguageMismatch,
                                page.id,
                                element.id,
                                Value::String(target_language.to_owned()),
                                Value::String(language.to_owned()),
                            ),
                            Some(_) => counts.target_language_translations += 1,
                        }
                    }
                }

                if element.final_scene.eligible {
                    counts.render_eligible_translations += 1;
                } else if page.render_error.is_none() {
                    reject(
                        &mut rejection_reasons,
                        AcceptanceRejectionCode::TranslationNotRenderEligible,
                        page.id,
                        element.id,
                        Value::String("text layer in rendered final scene".to_owned()),
                        Value::Bool(false),
                    );
                }
                if !element.visibility.effective_visible {
                    reject(
                        &mut rejection_reasons,
                        AcceptanceRejectionCode::TranslationHidden,
                        page.id,
                        element.id,
                        Value::Bool(true),
                        Value::Bool(false),
                    );
                }
                if element.visibility.effective_opacity <= 0.0 {
                    reject(
                        &mut rejection_reasons,
                        AcceptanceRejectionCode::TranslationZeroOpacity,
                        page.id,
                        element.id,
                        json!({ "exclusive_minimum": 0.0 }),
                        json!({ "value": element.visibility.effective_opacity }),
                    );
                }
                if !element.final_scene.geometry_visible {
                    reject(
                        &mut rejection_reasons,
                        AcceptanceRejectionCode::TranslationGeometryNotVisible,
                        page.id,
                        element.id,
                        Value::String(
                            "nonempty rendered glyph bounds intersecting the page".to_owned(),
                        ),
                        serde_json::to_value(element.final_scene.glyph_bounds)
                            .unwrap_or(Value::Null),
                    );
                }

                let measurements =
                    measure_element(element, page, source_language, target_language, thresholds);
                if validated_translated_decorative_sfx {
                    evaluate_decorative_sfx_layout(
                        element,
                        &measurements,
                        thresholds,
                        page.id,
                        element.id,
                        &mut rejection_reasons,
                    );
                } else {
                    evaluate_layout_measurements(
                        &measurements,
                        thresholds,
                        page.id,
                        element.id,
                        &mut rejection_reasons,
                    );
                }
                if element.final_scene.visible {
                    counts.visible_translations += 1;
                }
                let accepted =
                    rejection_reasons.len() == rejection_start && element.final_scene.visible;
                counts.accepted_elements += usize::from(accepted);
                elements.push(ElementAcceptance {
                    element_id: element.id,
                    logical_group_id: logical_group.map(|membership| membership.group_id),
                    render_owner_element_id,
                    evaluated_render_owner,
                    accepted,
                    measurements,
                });
                continue;
            }

            let accepted = rejection_reasons.len() == rejection_start;
            let measurements =
                measure_element(element, page, source_language, target_language, thresholds);
            counts.accepted_elements += usize::from(accepted);
            elements.push(ElementAcceptance {
                element_id: element.id,
                logical_group_id: logical_group.map(|membership| membership.group_id),
                render_owner_element_id,
                evaluated_render_owner,
                accepted,
                measurements,
            });
        }

        if counts.detected_source_elements == 0 {
            rejection_reasons.push(AcceptanceRejection {
                code: AcceptanceRejectionCode::PageNoDetectedSourceText,
                page_id: page.id,
                element_id: None,
                related_element_id: None,
                expected: Value::String("at least one detected source text element".to_owned()),
                actual: Value::from(0),
            });
        }

        let pairs = evaluate_pairs(page, thresholds, &mut rejection_reasons);
        counts.pairwise_conflicts = pairs.iter().filter(|pair| !pair.accepted).count();
        let accepted = counts.detected_source_elements > 0
            && counts.accepted_elements == counts.required_source_elements
            && page.render_error.is_none()
            && rejection_reasons.len() == page_rejection_start;
        pages.push(PageAcceptance {
            page_id: page.id,
            label: page.label.clone(),
            accepted,
            counts,
            elements,
            pairs,
        });
    }

    AcceptanceRecord {
        schema_version: ACCEPTANCE_SCHEMA_VERSION,
        accepted: !pages.is_empty()
            && pages.iter().all(|page| page.accepted)
            && rejection_reasons.is_empty(),
        source_language: source_language.to_owned(),
        target_language: target_language.to_owned(),
        thresholds: thresholds.clone(),
        pages,
        rejection_reasons,
        semantic_fidelity_requires_external_review: true,
    }
}

fn validate_decorative_sfx_decision(
    page: &PageInspection,
    element: &TextElementInspection,
    decision: &DecorativeSfxDecision,
    reasons: &mut Vec<AcceptanceRejection>,
) -> bool {
    let shared_valid = decision.schema_version == DECORATIVE_SFX_DECISION_SCHEMA_VERSION
        && element.detected
        && element.source_region_id == Some(decision.source_region_id)
        && element.source.as_ref().is_some_and(|source| {
            source.text == decision.source_ocr.text
                && source.language == decision.source_ocr.language
        })
        && !is_actual_container_bound(element)
        && decision.page_id == page.id
        && decision.element_id == element.id
        && decision.content_id == element.content_id
        && decision.original_ordinal > 0;
    let disposition_valid = match decision.disposition {
        DecorativeSfxDisposition::Translate => {
            element.text_role.as_deref() == Some(DECORATIVE_SFX_ROLE)
                && element.required
                && element.translation.is_some()
                && element.visibility.effective_visible
                && element.final_scene.visible
                && decision.review_state == "translated_decorative_sfx"
                && decision.target_translation_owner == Some(element.content_id)
                && decision.target_render_owner == Some(element.id)
        }
        DecorativeSfxDisposition::SkipDifficult => {
            element.text_role.as_deref() == Some(SKIPPED_DIFFICULT_SFX_ROLE)
                && !element.required
                && element.translation.is_none()
                && !element.visibility.effective_visible
                && !element.final_scene.visible
                && decision.review_state == "skipped_difficult_sfx"
                && decision.target_translation_owner.is_none()
                && decision.target_render_owner.is_none()
        }
        DecorativeSfxDisposition::RetainRequired => {
            matches!(
                element.text_role.as_deref(),
                Some(FREE_TEXT_ROLE | crate::free_dialogue::DIALOGUE_ROLE)
            ) && element.required
                && decision.review_state == "retained_required_non_decorative_sfx"
                && decision.target_translation_owner == Some(element.content_id)
                && decision.target_render_owner == Some(element.id)
        }
    };
    let valid = shared_valid && disposition_valid;
    if !valid {
        reject(
            reasons,
            AcceptanceRejectionCode::InvalidDecorativeSfxDecision,
            page.id,
            element.id,
            Value::String(
                "evidence-backed SFX-screening decision with source identity and disposition-consistent role, required state, translation, and render ownership"
                    .to_owned(),
            ),
            serde_json::to_value(decision).unwrap_or(Value::Null),
        );
    }
    valid
}

fn validate_logical_dialogue_groups(page: &PageInspection, reasons: &mut Vec<AcceptanceRejection>) {
    let mut group_ids = std::collections::BTreeSet::new();
    let mut owned_elements = std::collections::BTreeMap::<EntityId, EntityId>::new();
    for group in &page.logical_dialogue_groups {
        let structurally_ordered = group.members.len() >= 2
            && group.members.first().is_some_and(|member| {
                member.element_id == group.primary_render_element_id
                    && member.content_id == group.group_id
            })
            && group
                .members
                .iter()
                .enumerate()
                .all(|(index, member)| member.ordinal == index as u32 + 1)
            && group.logical_source_text
                == group
                    .members
                    .iter()
                    .map(|member| member.source_text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
        if !group_ids.insert(group.group_id) || !structurally_ordered {
            reject(
                reasons,
                AcceptanceRejectionCode::LogicalDialogueMemberMismatch,
                page.id,
                group.primary_render_element_id,
                Value::String("one unique group with at least two sequential members and the first member as render owner".to_owned()),
                serde_json::to_value(group).unwrap_or(Value::Null),
            );
        }
        for member in &group.members {
            if let Some(previous) = owned_elements.insert(member.element_id, group.group_id) {
                reasons.push(AcceptanceRejection {
                    code: AcceptanceRejectionCode::LogicalDialogueOwnershipConflict,
                    page_id: page.id,
                    element_id: Some(member.element_id),
                    related_element_id: Some(group.primary_render_element_id),
                    expected: Value::String("exactly one logical dialogue owner".to_owned()),
                    actual: json!({ "group_ids": [previous, group.group_id] }),
                });
            }
            let matching = page.text_elements.iter().find(|element| {
                element.id == member.element_id
                    && element.content_id == member.content_id
                    && element.source_region_id == Some(member.source_region_id)
                    && element.required
                    && element.source.as_ref().map(|source| source.text.as_str())
                        == Some(member.source_text.as_str())
                    && element
                        .source
                        .as_ref()
                        .and_then(|source| source.language.as_deref())
                        .is_some_and(is_japanese)
                    && element.source_writing_mode == Some(WritingMode::Vertical)
                    && element.logical_dialogue_memberships.len() == 1
                    && element.logical_dialogue_memberships[0].group_id == group.group_id
                    && element.logical_dialogue_memberships[0].primary_render_element_id
                        == group.primary_render_element_id
                    && element.logical_dialogue_memberships[0].target_region_id
                        == group.target_region_id
                    && element.logical_dialogue_memberships[0].member_ordinal == member.ordinal
                    && element.text_safe_region.as_ref().is_some_and(|region| {
                        region.id == group.target_region_id
                            && valid_geometry(&region.geometry)
                            && element.source_geometry.as_ref().is_some_and(|source| {
                                valid_geometry(source)
                                    && intersection_area(source.bounds, region.geometry.bounds)
                                        / area(source.bounds)
                                        >= RELIABLE_TARGET_REGION_CONTAINMENT
                            })
                            && region.association.is_some_and(|association| {
                                association.layout_relation == TargetLayoutRelation::FlowsIn
                                    && association.source_inside_target_relation
                            })
                    })
            });
            if matching.is_none() {
                reasons.push(AcceptanceRejection {
                    code: AcceptanceRejectionCode::LogicalDialogueMemberMismatch,
                    page_id: page.id,
                    element_id: Some(group.primary_render_element_id),
                    related_element_id: Some(member.element_id),
                    expected: json!({
                        "group_id": group.group_id,
                        "target_region_id": group.target_region_id,
                        "member": member,
                    }),
                    actual: Value::String("required source member or verified target association is missing or changed".to_owned()),
                });
            }
        }
    }
}

fn validate_element_logical_ownership<'a>(
    page: &PageInspection,
    element: &'a TextElementInspection,
    reasons: &mut Vec<AcceptanceRejection>,
) -> Option<&'a LogicalDialogueMembershipInspection> {
    let Some(membership) = element.logical_dialogue_memberships.first() else {
        return None;
    };
    if element.logical_dialogue_memberships.len() != 1 {
        reject(
            reasons,
            AcceptanceRejectionCode::LogicalDialogueOwnershipConflict,
            page.id,
            element.id,
            Value::String("exactly one logical dialogue owner".to_owned()),
            serde_json::to_value(&element.logical_dialogue_memberships).unwrap_or(Value::Null),
        );
        return None;
    }
    let valid = page.logical_dialogue_groups.iter().any(|group| {
        group.group_id == membership.group_id
            && group.primary_render_element_id == membership.primary_render_element_id
            && group.target_region_id == membership.target_region_id
            && group.members.iter().any(|member| {
                member.element_id == element.id
                    && member.content_id == element.content_id
                    && Some(member.source_region_id) == element.source_region_id
                    && member.ordinal == membership.member_ordinal
            })
    });
    if !valid {
        reject(
            reasons,
            AcceptanceRejectionCode::LogicalDialogueMemberMismatch,
            page.id,
            element.id,
            Value::String("membership matching the page logical dialogue provenance".to_owned()),
            serde_json::to_value(membership).unwrap_or(Value::Null),
        );
        None
    } else {
        Some(membership)
    }
}

fn measure_element(
    element: &TextElementInspection,
    page: &PageInspection,
    source_language: &str,
    target_language: &str,
    thresholds: &QualityThresholds,
) -> ElementMeasurements {
    let raw_source_bounds = element
        .source_geometry
        .as_ref()
        .map(|geometry| geometry.bounds);
    let raw_glyph_bounds = element.final_scene.glyph_bounds;
    let raw_layout_bounds = element.final_scene.layout_bounds;
    let raw_font_size_px = element.final_scene.font_size_px;
    let source_bounds = raw_source_bounds.filter(|bounds| finite_bounds(*bounds));
    let glyph_bounds = raw_glyph_bounds.filter(|bounds| finite_bounds(*bounds));
    let layout_bounds = raw_layout_bounds.filter(|bounds| finite_bounds(*bounds));
    let font_size_px = raw_font_size_px.filter(|value| value.is_finite());
    let finite_positive_layout = raw_source_bounds.is_some_and(valid_bounds)
        && raw_glyph_bounds.is_some_and(valid_bounds)
        && raw_layout_bounds.is_some_and(valid_bounds)
        && raw_font_size_px.is_some_and(|value| value.is_finite() && value > 0.0)
        && element.final_scene.diagnostics.iter().all(|diagnostic| {
            matches!(
                diagnostic.as_str(),
                "text_overflow" | "text_below_readable_size"
            )
        });
    let mut invalid_metrics = Vec::new();
    if raw_source_bounds.is_some() && source_bounds.is_none() {
        invalid_metrics.push("source_region_bounds_non_finite".to_owned());
    }
    if raw_glyph_bounds.is_some() && glyph_bounds.is_none() {
        invalid_metrics.push("rendered_glyph_bounds_non_finite".to_owned());
    }
    if raw_layout_bounds.is_some() && layout_bounds.is_none() {
        invalid_metrics.push("layout_bounds_non_finite".to_owned());
    }
    if raw_font_size_px.is_some() && font_size_px.is_none() {
        invalid_metrics.push("rendered_font_size_non_finite".to_owned());
    }
    let source_region_area_coverage = source_bounds
        .zip(glyph_bounds)
        .filter(|(source, glyph)| valid_bounds(*source) && valid_bounds(*glyph))
        .map(|(source, glyph)| intersection_area(source, glyph) / area(source));
    let source_region_overflow_px = source_bounds
        .zip(glyph_bounds)
        .filter(|(source, glyph)| valid_bounds(*source) && valid_bounds(*glyph))
        .map(|(source, glyph)| overflow_px(glyph, source));
    let target_layout_anchor = measure_layout_anchor(
        element,
        page.id,
        source_bounds,
        glyph_bounds,
        source_language,
        target_language,
        thresholds,
    );
    let maximum_reasonable_line_count = target_layout_anchor
        .expected_text_units
        .max(1)
        .min(MAX_REASONABLE_RENDERED_LINE_COUNT);
    let rendered_line_count_reasonable = element
        .final_scene
        .line_count
        .is_some_and(|count| (1..=maximum_reasonable_line_count).contains(&count));
    let page_bounds = ElementBounds {
        x: 0.0,
        y: 0.0,
        width: page.width,
        height: page.height,
    };
    let page_overflow_px = glyph_bounds
        .filter(|glyph| valid_bounds(*glyph) && valid_bounds(page_bounds))
        .map(|glyph| overflow_px(glyph, page_bounds));
    let rejected_target_anchor_claim = (element.verified_ui_panel_anchor.is_some()
        || element.verified_free_dialogue_anchor.is_some())
        && target_layout_anchor.kind == LayoutAnchorKind::SourceRegion;
    let text_safe_containment = if rejected_target_anchor_claim {
        element
            .source_geometry
            .as_ref()
            .zip(element.source_region_id)
            .map(|(geometry, id)| TextSafeRegion {
                id,
                kind: element
                    .source_region_kind
                    .clone()
                    .unwrap_or_else(|| "dev.koharu.region.text".to_owned()),
                geometry: geometry.clone(),
                association: None,
            })
            .as_ref()
            .map(|region| {
                measure_text_safe_containment(
                    region,
                    element.final_scene.glyph_ink.as_ref(),
                    thresholds.min_text_safe_padding_px,
                    true,
                )
            })
    } else {
        element.text_safe_region.as_ref().map(|region| {
            measure_text_safe_containment(
                region,
                element.final_scene.glyph_ink.as_ref(),
                thresholds.min_text_safe_padding_px,
                target_layout_anchor.kind != LayoutAnchorKind::AdjacentFreeDialogueTextSafeAnchor,
            )
        })
    };
    ElementMeasurements {
        requested_source_language: source_language.to_owned(),
        actual_source_language: element
            .source
            .as_ref()
            .and_then(|source| source.language.clone()),
        source_language_matches: element
            .source
            .as_ref()
            .and_then(|source| source.language.as_deref())
            == Some(source_language),
        source_region_bounds: source_bounds,
        rendered_glyph_bounds: glyph_bounds,
        layout_bounds,
        rendered_font_size_px: font_size_px,
        rendered_line_count: element.final_scene.line_count,
        maximum_reasonable_line_count,
        rendered_line_count_reasonable,
        rendered_glyph_height_px: glyph_bounds.map(|bounds| bounds.height),
        source_region_area_coverage,
        source_region_overflow_px,
        target_layout_anchor,
        free_dialogue_anchor_evidence: element.verified_free_dialogue_anchor.clone(),
        text_safe_containment,
        page_overflow_px,
        finite_positive_layout,
        renderer_diagnostics: element.final_scene.diagnostics.clone(),
        invalid_metrics,
    }
}

fn measure_layout_anchor(
    element: &TextElementInspection,
    page_id: EntityId,
    source_bounds: Option<ElementBounds>,
    glyph_bounds: Option<ElementBounds>,
    requested_source_language: &str,
    requested_target_language: &str,
    thresholds: &QualityThresholds,
) -> LayoutAnchorMeasurement {
    let target_writing_mode = element
        .typography
        .as_ref()
        .and_then(|typography| typography.writing_mode);
    let transformed_languages = element
        .source
        .as_ref()
        .and_then(|text| text.language.as_deref())
        .is_some_and(is_japanese)
        && element
            .translation
            .as_ref()
            .and_then(|text| text.language.as_deref())
            .is_some_and(is_korean)
        && is_japanese(requested_source_language)
        && is_korean(requested_target_language);
    let transformed_writing_mode = element.source_writing_mode == Some(WritingMode::Vertical)
        && target_writing_mode == Some(WritingMode::Horizontal);
    let candidate = element
        .text_safe_region
        .as_ref()
        .filter(|_| is_actual_container_bound(element));
    let association_confidence = candidate
        .and_then(|region| source_bounds.map(|source| (source, region.geometry.bounds)))
        .filter(|(source, target)| valid_bounds(*source) && valid_bounds(*target))
        .map(|(source, target)| intersection_area(source, target) / area(source));

    let verified_ui_panel = element
        .verified_ui_panel_anchor
        .as_ref()
        .filter(|decision| {
            element.required
                && element
                    .decorative_sfx
                    .as_ref()
                    .is_none_or(DecorativeSfxDecision::requires_translation)
                && element.text_role.as_deref() == Some(UI_TEXT_ROLE)
                && element.logical_dialogue_memberships.is_empty()
                && decision.element_id == element.id
                && decision.page_id == page_id
                && decision.content_id == element.content_id
                && Some(decision.source_region_id) == element.source_region_id
                && candidate.is_some_and(|region| {
                    region.id == decision.panel.region_id
                        && geometry_equivalent(&region.geometry, &decision.panel.geometry)
                })
                && decision.panel.region_kind == koharu_scene::PanelRegion::KIND
                && element.source_geometry.as_ref().is_some_and(|source| {
                    let evidence = &decision.panel.raster_evidence;
                    evidence.panel_bbox == decision.panel.geometry.bounds
                        && raster_candidate_supports_ui_role(
                            evidence,
                            decision.source_region_id,
                            source.bounds,
                            &decision.evidence.ui_role_and_function,
                        )
                        .is_ok()
                })
                && decision.source_containment_ratio >= RELIABLE_TARGET_REGION_CONTAINMENT
                && association_confidence
                    .is_some_and(|value| value >= RELIABLE_TARGET_REGION_CONTAINMENT)
                && decision.confidence >= crate::ui_panel::MINIMUM_UI_PANEL_VERIFICATION_CONFIDENCE
                && !decision.association_reason.trim().is_empty()
        });
    let verified_free_dialogue =
        element
            .verified_free_dialogue_anchor
            .as_ref()
            .filter(|decision| {
                element.required
                    && element
                        .decorative_sfx
                        .as_ref()
                        .is_none_or(DecorativeSfxDecision::requires_translation)
                    && element.logical_dialogue_memberships.is_empty()
                    && element.text_role.as_deref()
                        == Some(decision.role_gate.source_scene_role.as_str())
                    && decision.element_id == element.id
                    && decision.page_id == page_id
                    && decision.content_id == element.content_id
                    && Some(decision.source_region_id) == element.source_region_id
                    && candidate.is_some_and(|region| {
                        region.id == decision.target_region_id
                            && region.kind == koharu_scene::TextRegion::KIND
                            && region.geometry.bounds == decision.candidate_bounds
                    })
                    && element.source_geometry.as_ref().is_some_and(|source| {
                        element.source.as_ref().is_some_and(|source_text| {
                            element.translation.as_ref().is_some_and(|target_text| {
                                validate_free_dialogue_anchor_decision(
                                    decision,
                                    element.id,
                                    element.content_id,
                                    decision.source_region_id,
                                    source.bounds,
                                    &source_text.text,
                                    &target_text.text,
                                )
                                .is_ok()
                            })
                        })
                    })
                    && !decision.association_reason.trim().is_empty()
            });

    let (association_reason, reliable_region, ui_panel_anchor, free_dialogue_anchor) =
        match candidate {
            None => ("no_explicit_text_safe_target_relation", None, false, false),
            Some(_) if verified_ui_panel.is_some() && !transformed_languages => (
                "ui_panel_source_target_languages_are_not_japanese_to_korean",
                None,
                false,
                false,
            ),
            Some(region) if verified_ui_panel.is_some() && !valid_geometry(&region.geometry) => (
                "ui_panel_geometry_is_not_finite_and_positive",
                None,
                false,
                false,
            ),
            Some(_)
                if verified_ui_panel.is_some()
                    && !element.source_geometry.as_ref().is_some_and(valid_geometry) =>
            {
                (
                    "source_region_geometry_is_not_finite_and_positive",
                    None,
                    false,
                    false,
                )
            }
            Some(region) if let Some(decision) = verified_ui_panel => {
                let safe_interior = decision.panel.raster_evidence.safe_interior_bbox;
                match valid_bounds(safe_interior) {
                    true => (
                        decision.association_reason.as_str(),
                        Some((region, safe_interior)),
                        true,
                        false,
                    ),
                    false => (
                        "detector_ui_panel_safe_interior_is_not_finite_and_positive",
                        None,
                        false,
                        false,
                    ),
                }
            }
            Some(_) if verified_free_dialogue.is_some() && !transformed_languages => (
                "free_dialogue_source_target_languages_are_not_japanese_to_korean",
                None,
                false,
                false,
            ),
            Some(_) if verified_free_dialogue.is_some() && !transformed_writing_mode => (
                "free_dialogue_source_target_writing_modes_are_not_vertical_to_horizontal",
                None,
                false,
                false,
            ),
            Some(region)
                if verified_free_dialogue.is_some() && !valid_geometry(&region.geometry) =>
            {
                (
                    "free_dialogue_anchor_geometry_is_not_finite_and_positive",
                    None,
                    false,
                    false,
                )
            }
            Some(region) if let Some(decision) = verified_free_dialogue => {
                match inset_bounds(region.geometry.bounds, thresholds.min_text_safe_padding_px) {
                    Some(bounds) => (
                        decision.association_reason.as_str(),
                        Some((region, bounds)),
                        false,
                        true,
                    ),
                    None => (
                        "inset_free_dialogue_anchor_is_not_finite_and_positive",
                        None,
                        false,
                        false,
                    ),
                }
            }
            Some(_) if !transformed_languages => (
                "source_target_languages_are_not_japanese_to_korean",
                None,
                false,
                false,
            ),
            Some(_) if !transformed_writing_mode => (
                "source_target_writing_modes_are_not_vertical_to_horizontal",
                None,
                false,
                false,
            ),
            Some(region) if !valid_geometry(&region.geometry) => (
                "target_region_geometry_is_not_finite_and_positive",
                None,
                false,
                false,
            ),
            Some(_) if !element.source_geometry.as_ref().is_some_and(valid_geometry) => (
                "source_region_geometry_is_not_finite_and_positive",
                None,
                false,
                false,
            ),
            Some(region)
                if !region.association.is_some_and(|association| {
                    association.layout_relation == TargetLayoutRelation::FlowsIn
                        && association.source_inside_target_relation
                }) =>
            {
                (
                    "flows_in_and_source_inside_target_relations_are_required",
                    None,
                    false,
                    false,
                )
            }
            Some(_)
                if association_confidence
                    .is_none_or(|value| value < RELIABLE_TARGET_REGION_CONTAINMENT) =>
            {
                (
                    "source_region_containment_in_target_is_below_reliable_minimum",
                    None,
                    false,
                    false,
                )
            }
            Some(region) => {
                match inset_bounds(region.geometry.bounds, thresholds.min_text_safe_padding_px) {
                    Some(bounds) => (
                        "verified_japanese_vertical_to_korean_horizontal_text_safe_target",
                        Some((region, bounds)),
                        false,
                        false,
                    ),
                    None => (
                        "inset_target_text_safe_geometry_is_not_finite_and_positive",
                        None,
                        false,
                        false,
                    ),
                }
            }
        };

    let (kind, region_id, region_kind, bounds) = reliable_region.map_or_else(
        || {
            (
                LayoutAnchorKind::SourceRegion,
                element.source_region_id,
                element.source_region_kind.clone(),
                source_bounds,
            )
        },
        |(region, bounds)| {
            (
                if ui_panel_anchor {
                    LayoutAnchorKind::UiPanelTextSafeInterior
                } else if free_dialogue_anchor {
                    LayoutAnchorKind::AdjacentFreeDialogueTextSafeAnchor
                } else {
                    LayoutAnchorKind::TextSafeRegion
                },
                Some(region.id),
                Some(region.kind.clone()),
                Some(bounds),
            )
        },
    );
    let rendered_area_coverage = bounds
        .zip(glyph_bounds)
        .filter(|(anchor, glyph)| valid_bounds(*anchor) && valid_bounds(*glyph))
        .map(|(anchor, glyph)| intersection_area(anchor, glyph) / area(anchor));
    let expected_text_units = element
        .translation
        .as_ref()
        .map_or(0, |translation| visible_text_units(&translation.text));
    let target_anchor_area_px2 = bounds.filter(|bounds| valid_bounds(*bounds)).map(area);
    let expected_ink_area_px2 = expected_text_units as f64
        * thresholds.min_rendered_font_size_px
        * thresholds.min_rendered_glyph_height_px
        * EXPECTED_VISIBLE_UNIT_AREA_FACTOR;
    let measured_ink_bounds_area_px2 = bounds
        .zip(glyph_bounds)
        .filter(|(anchor, glyph)| valid_bounds(*anchor) && valid_bounds(*glyph))
        .map(|(anchor, glyph)| intersection_area(anchor, glyph));
    let expected_area_coverage =
        target_anchor_area_px2.map(|anchor_area| expected_ink_area_px2 / anchor_area);
    let coverage_policy = match kind {
        LayoutAnchorKind::SourceRegion => LayoutAnchorCoveragePolicy::ConfiguredSourceAnchorFloor,
        LayoutAnchorKind::TextSafeRegion
        | LayoutAnchorKind::UiPanelTextSafeInterior
        | LayoutAnchorKind::AdjacentFreeDialogueTextSafeAnchor
            if expected_area_coverage
                .is_some_and(|coverage| coverage >= thresholds.min_layout_anchor_area_coverage) =>
        {
            LayoutAnchorCoveragePolicy::ConfiguredLongDialogueFloor
        }
        LayoutAnchorKind::TextSafeRegion
        | LayoutAnchorKind::UiPanelTextSafeInterior
        | LayoutAnchorKind::AdjacentFreeDialogueTextSafeAnchor => {
            LayoutAnchorCoveragePolicy::VerifiedTargetExpectedInk
        }
    };
    let required_area_coverage = target_anchor_area_px2.map(|anchor_area| match coverage_policy {
        LayoutAnchorCoveragePolicy::ConfiguredSourceAnchorFloor
        | LayoutAnchorCoveragePolicy::ConfiguredLongDialogueFloor => {
            thresholds.min_layout_anchor_area_coverage
        }
        LayoutAnchorCoveragePolicy::VerifiedTargetExpectedInk => (expected_ink_area_px2
            / anchor_area)
            .max(MIN_VERIFIED_TARGET_AREA_COVERAGE)
            .min(thresholds.min_layout_anchor_area_coverage),
    });
    let rendered_overflow_px = bounds
        .zip(glyph_bounds)
        .filter(|(anchor, glyph)| valid_bounds(*anchor) && valid_bounds(*glyph))
        .map(|(anchor, glyph)| overflow_px(glyph, anchor));
    let recorded_association_confidence = if free_dialogue_anchor {
        element
            .verified_free_dialogue_anchor
            .as_ref()
            .map(|decision| decision.association_confidence)
    } else {
        association_confidence
    };

    LayoutAnchorMeasurement {
        kind,
        region_id,
        region_kind,
        bounds,
        source_writing_mode: element.source_writing_mode,
        target_writing_mode,
        association_confidence: recorded_association_confidence,
        association_reason: association_reason.to_owned(),
        coverage_policy,
        expected_text_units,
        target_anchor_area_px2,
        expected_ink_area_px2,
        measured_ink_bounds_area_px2,
        rendered_area_coverage,
        required_area_coverage,
        rendered_overflow_px,
    }
}

fn visible_text_units(text: &str) -> usize {
    text.graphemes(true)
        .filter(|grapheme| {
            grapheme
                .chars()
                .any(|character| !character.is_whitespace() && !character.is_control())
        })
        .count()
}

fn is_japanese(language: &str) -> bool {
    language
        .split(['-', '_'])
        .next()
        .is_some_and(|primary| primary.eq_ignore_ascii_case("ja"))
}

fn is_korean(language: &str) -> bool {
    language
        .split(['-', '_'])
        .next()
        .is_some_and(|primary| primary.eq_ignore_ascii_case("ko"))
}

fn valid_geometry(geometry: &ElementGeometry) -> bool {
    valid_bounds(geometry.bounds)
        && geometry.points.len() >= 3
        && geometry
            .points
            .iter()
            .all(|point| point.x.is_finite() && point.y.is_finite())
}

fn geometry_equivalent(first: &ElementGeometry, second: &ElementGeometry) -> bool {
    first.points.len() == second.points.len()
        && first
            .points
            .iter()
            .zip(&second.points)
            .all(|(first, second)| first.x == second.x && first.y == second.y)
        && first.bounds == second.bounds
}

fn inset_bounds(bounds: ElementBounds, padding: f64) -> Option<ElementBounds> {
    let inset = ElementBounds {
        x: bounds.x + padding,
        y: bounds.y + padding,
        width: bounds.width - padding * 2.0,
        height: bounds.height - padding * 2.0,
    };
    valid_bounds(inset).then_some(inset)
}

fn evaluate_decorative_sfx_layout(
    element: &TextElementInspection,
    measurements: &ElementMeasurements,
    thresholds: &QualityThresholds,
    page_id: EntityId,
    element_id: EntityId,
    reasons: &mut Vec<AcceptanceRejection>,
) {
    let finite_positive_layout = measurements.source_region_bounds.is_some_and(valid_bounds)
        && measurements.rendered_glyph_bounds.is_some_and(valid_bounds)
        && measurements.layout_bounds.is_some_and(valid_bounds)
        && measurements
            .rendered_font_size_px
            .is_some_and(|value| value.is_finite() && value > 0.0)
        && measurements
            .renderer_diagnostics
            .iter()
            .all(|diagnostic| diagnostic == "text_overflow");
    if !finite_positive_layout {
        reject(
            reasons,
            AcceptanceRejectionCode::UnsupportedLayout,
            page_id,
            element_id,
            Value::String(
                "finite positive source, layout, glyph, and font metrics with no renderer diagnostics other than source-box text_overflow"
                    .to_owned(),
            ),
            serde_json::to_value(measurements).unwrap_or(Value::Null),
        );
    }

    let rasterized_glyph_ink = element
        .final_scene
        .glyph_ink
        .as_ref()
        .is_some_and(|mask| mask.valid() && mask.alpha.iter().any(|alpha| *alpha > 0));
    if !rasterized_glyph_ink {
        reject(
            reasons,
            AcceptanceRejectionCode::RenderedGlyphInkMissing,
            page_id,
            element_id,
            Value::String("nonempty rasterized glyph ink".to_owned()),
            Value::Null,
        );
    }

    if !element.visibility.effective_opacity.is_finite()
        || element.visibility.effective_opacity <= 0.0
        || !element.final_scene.opacity.is_finite()
        || element.final_scene.opacity <= 0.0
    {
        reject(
            reasons,
            AcceptanceRejectionCode::TranslationZeroOpacity,
            page_id,
            element_id,
            json!({ "finite_exclusive_minimum": 0.0 }),
            json!({
                "effective_opacity": element.visibility.effective_opacity,
                "final_scene_opacity": element.final_scene.opacity,
            }),
        );
    }

    match measurements.page_overflow_px {
        Some(value) if value <= thresholds.max_page_overflow_px => {}
        value => reject(
            reasons,
            AcceptanceRejectionCode::RenderedTextOutsidePage,
            page_id,
            element_id,
            json!({ "maximum_px": thresholds.max_page_overflow_px }),
            json!({ "value_px": value }),
        ),
    }
}

fn evaluate_layout_measurements(
    measurements: &ElementMeasurements,
    thresholds: &QualityThresholds,
    page_id: EntityId,
    element_id: EntityId,
    reasons: &mut Vec<AcceptanceRejection>,
) {
    if measurements.source_region_bounds.is_none() {
        reject(
            reasons,
            AcceptanceRejectionCode::SourceRegionGeometryMissing,
            page_id,
            element_id,
            Value::String("detected source-region geometry".to_owned()),
            Value::Null,
        );
    }
    match &measurements.text_safe_containment {
        None => reject(
            reasons,
            AcceptanceRejectionCode::TextSafeRegionMissing,
            page_id,
            element_id,
            Value::String("detected balloon, verified UI panel, or text-region contour".to_owned()),
            Value::Null,
        ),
        Some(measurement) if measurement.minimum_clearance_px.is_none() => reject(
            reasons,
            AcceptanceRejectionCode::RenderedGlyphInkMissing,
            page_id,
            element_id,
            Value::String("nonempty rasterized glyph ink".to_owned()),
            serde_json::to_value(measurement).unwrap_or(Value::Null),
        ),
        Some(_) => {}
    }
    if !measurements.finite_positive_layout {
        reject(reasons, AcceptanceRejectionCode::UnsupportedLayout, page_id, element_id, Value::String("finite positive source, layout, glyph, and font metrics with no renderer failure diagnostics".to_owned()), serde_json::to_value(measurements).unwrap_or(Value::Null));
    }
    match measurements.rendered_font_size_px {
        None => reject(
            reasons,
            AcceptanceRejectionCode::RenderedFontSizeMissing,
            page_id,
            element_id,
            Value::String("finite positive rendered font size".to_owned()),
            Value::Null,
        ),
        Some(_) => {}
    }
    match measurements.rendered_line_count {
        None => reject(
            reasons,
            AcceptanceRejectionCode::RenderedLineCountMissing,
            page_id,
            element_id,
            json!({
                "minimum": 1,
                "maximum": measurements.maximum_reasonable_line_count,
            }),
            Value::Null,
        ),
        Some(value) if !measurements.rendered_line_count_reasonable => reject(
            reasons,
            AcceptanceRejectionCode::RenderedLineCountUnreasonable,
            page_id,
            element_id,
            json!({
                "minimum": 1,
                "maximum": measurements.maximum_reasonable_line_count,
                "basis": "at least one visible target grapheme per line and at most 12 lines",
                "expected_text_units": measurements.target_layout_anchor.expected_text_units,
            }),
            json!({ "value": value }),
        ),
        Some(_) => {}
    }
    if let Some(value) = measurements.page_overflow_px
        && value > thresholds.max_page_overflow_px
    {
        reject(
            reasons,
            AcceptanceRejectionCode::RenderedTextOutsidePage,
            page_id,
            element_id,
            json!({ "maximum_px": thresholds.max_page_overflow_px }),
            json!({ "value_px": value }),
        );
    }
}

fn measure_text_safe_containment(
    region: &TextSafeRegion,
    glyph_ink: Option<&GlyphInkMask>,
    required_padding_px: f64,
    include_pixel_footprint: bool,
) -> TextSafeContainment {
    let mut measurement = TextSafeContainment {
        region_id: region.id,
        region_kind: region.kind.clone(),
        region_bounds: region.geometry.bounds,
        required_padding_px,
        minimum_clearance_px: None,
        glyph_ink_pixels: 0,
        violation_pixels: 0,
        violations: Vec::new(),
    };
    let Some(mask) = glyph_ink.filter(|mask| mask.valid()) else {
        return measurement;
    };
    let polygon = &region.geometry.points;
    if polygon.len() < 3
        || polygon
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        return measurement;
    }

    for (index, &alpha) in mask
        .alpha
        .iter()
        .enumerate()
        .filter(|(_, alpha)| **alpha > 0)
    {
        let x = index % mask.width as usize;
        let y = index / mask.width as usize;
        let point = ElementPoint {
            x: f64::from(mask.left) + x as f64 + 0.5,
            y: f64::from(mask.top) + y as f64 + 0.5,
        };
        // Detector-owned adjacent free-dialogue anchors already expose a raster-sampled 4 px
        // interior. Charging the sample's half-diagonal again silently turns that exact gate into
        // 4.707 px and makes its declared 12 px glyph room impossible. Other contour owners retain
        // the conservative whole-pixel footprint measurement.
        let pixel_footprint = if include_pixel_footprint {
            PIXEL_HALF_DIAGONAL
        } else {
            0.0
        };
        let clearance = signed_distance_to_polygon(point, polygon) - pixel_footprint;
        measurement.glyph_ink_pixels += 1;
        measurement.minimum_clearance_px = Some(
            measurement
                .minimum_clearance_px
                .map_or(clearance, |minimum| minimum.min(clearance)),
        );
        if clearance < required_padding_px {
            measurement.violation_pixels += 1;
            record_worst_violation(
                &mut measurement.violations,
                TextSafeViolation {
                    x: point.x,
                    y: point.y,
                    alpha,
                    clearance_px: clearance,
                },
            );
        }
    }
    measurement
}

impl GlyphInkMask {
    fn valid(&self) -> bool {
        self.width > 0
            && self.height > 0
            && usize::try_from(u64::from(self.width) * u64::from(self.height))
                .is_ok_and(|pixels| pixels == self.alpha.len())
    }
}

fn record_worst_violation(violations: &mut Vec<TextSafeViolation>, value: TextSafeViolation) {
    let index = violations.partition_point(|candidate| {
        candidate
            .clearance_px
            .total_cmp(&value.clearance_px)
            .is_le()
    });
    if index < MAX_RECORDED_TEXT_SAFE_VIOLATIONS {
        violations.insert(index, value);
        violations.truncate(MAX_RECORDED_TEXT_SAFE_VIOLATIONS);
    }
}

fn signed_distance_to_polygon(point: ElementPoint, polygon: &[ElementPoint]) -> f64 {
    let mut inside = false;
    let mut minimum_squared = f64::INFINITY;
    for index in 0..polygon.len() {
        let first = polygon[index];
        let second = polygon[(index + 1) % polygon.len()];
        minimum_squared = minimum_squared.min(distance_to_segment_squared(point, first, second));
        if (first.y > point.y) != (second.y > point.y) {
            let crossing_x =
                (second.x - first.x) * (point.y - first.y) / (second.y - first.y) + first.x;
            inside ^= point.x < crossing_x;
        }
    }
    let distance = minimum_squared.sqrt();
    if inside { distance } else { -distance }
}

fn distance_to_segment_squared(
    point: ElementPoint,
    first: ElementPoint,
    second: ElementPoint,
) -> f64 {
    let dx = second.x - first.x;
    let dy = second.y - first.y;
    let length_squared = dx * dx + dy * dy;
    if length_squared <= f64::EPSILON {
        return (point.x - first.x).powi(2) + (point.y - first.y).powi(2);
    }
    let projection = ((point.x - first.x) * dx + (point.y - first.y) * dy) / length_squared;
    let projection = projection.clamp(0.0, 1.0);
    let nearest_x = first.x + projection * dx;
    let nearest_y = first.y + projection * dy;
    (point.x - nearest_x).powi(2) + (point.y - nearest_y).powi(2)
}

fn evaluate_pairs(
    page: &PageInspection,
    thresholds: &QualityThresholds,
    reasons: &mut Vec<AcceptanceRejection>,
) -> Vec<PairAcceptance> {
    let required = page
        .text_elements
        .iter()
        .filter(|element| {
            element.required
                && element
                    .logical_dialogue_memberships
                    .first()
                    .is_none_or(|membership| {
                        element.logical_dialogue_memberships.len() == 1
                            && membership.primary_render_element_id == element.id
                    })
        })
        .collect::<Vec<_>>();
    let mut pairs = Vec::new();
    for (index, first) in required.iter().enumerate() {
        for second in required.iter().skip(index + 1) {
            let first_text = first
                .translation
                .as_ref()
                .map(|translation| normalize_text(&translation.text));
            let second_text = second
                .translation
                .as_ref()
                .map(|translation| normalize_text(&translation.text));
            let normalized_translation_equal = first_text
                .zip(second_text)
                .is_some_and(|(first, second)| !first.is_empty() && first == second);
            let rendered_region_overlap = first
                .final_scene
                .glyph_bounds
                .zip(second.final_scene.glyph_bounds)
                .and_then(overlap_over_smaller);
            let source_region_overlap = first
                .source_geometry
                .as_ref()
                .map(|geometry| geometry.bounds)
                .zip(
                    second
                        .source_geometry
                        .as_ref()
                        .map(|geometry| geometry.bounds),
                )
                .and_then(overlap_over_smaller);
            let mut accepted = true;
            if rendered_region_overlap
                .is_some_and(|value| value > thresholds.max_translated_region_overlap)
            {
                accepted = false;
                reject_pair(
                    reasons,
                    AcceptanceRejectionCode::TranslatedRegionOverlap,
                    page.id,
                    first.id,
                    second.id,
                    json!({ "maximum_ratio": thresholds.max_translated_region_overlap }),
                    json!({ "value_ratio": rendered_region_overlap }),
                );
            }
            if source_region_overlap
                .is_some_and(|value| value > thresholds.max_source_region_overlap)
            {
                accepted = false;
                reject_pair(
                    reasons,
                    AcceptanceRejectionCode::SourceRegionOverlap,
                    page.id,
                    first.id,
                    second.id,
                    json!({ "maximum_ratio": thresholds.max_source_region_overlap }),
                    json!({ "value_ratio": source_region_overlap }),
                );
            }
            if normalized_translation_equal
                && source_region_overlap
                    .is_some_and(|value| value > thresholds.max_duplicate_source_region_overlap)
            {
                accepted = false;
                reject_pair(
                    reasons,
                    AcceptanceRejectionCode::DuplicateTranslationRegion,
                    page.id,
                    first.id,
                    second.id,
                    json!({ "normalized_translation_equal": false, "maximum_source_overlap_ratio": thresholds.max_duplicate_source_region_overlap }),
                    json!({ "normalized_translation_equal": true, "source_overlap_ratio": source_region_overlap }),
                );
            }
            pairs.push(PairAcceptance {
                first_element_id: first.id,
                second_element_id: second.id,
                accepted,
                normalized_translation_equal,
                rendered_region_overlap,
                source_region_overlap,
            });
        }
    }
    pairs
}

fn reject(
    reasons: &mut Vec<AcceptanceRejection>,
    code: AcceptanceRejectionCode,
    page_id: EntityId,
    element_id: EntityId,
    expected: Value,
    actual: Value,
) {
    reasons.push(AcceptanceRejection {
        code,
        page_id,
        element_id: Some(element_id),
        related_element_id: None,
        expected,
        actual,
    });
}

fn reject_pair(
    reasons: &mut Vec<AcceptanceRejection>,
    code: AcceptanceRejectionCode,
    page_id: EntityId,
    element_id: EntityId,
    related_element_id: EntityId,
    expected: Value,
    actual: Value,
) {
    reasons.push(AcceptanceRejection {
        code,
        page_id,
        element_id: Some(element_id),
        related_element_id: Some(related_element_id),
        expected,
        actual,
    });
}

fn valid_bounds(bounds: ElementBounds) -> bool {
    finite_bounds(bounds) && bounds.width > 0.0 && bounds.height > 0.0
}

fn finite_bounds(bounds: ElementBounds) -> bool {
    [bounds.x, bounds.y, bounds.width, bounds.height]
        .into_iter()
        .all(f64::is_finite)
}

fn area(bounds: ElementBounds) -> f64 {
    bounds.width * bounds.height
}

fn intersection_area(first: ElementBounds, second: ElementBounds) -> f64 {
    let width = (first.x + first.width).min(second.x + second.width) - first.x.max(second.x);
    let height = (first.y + first.height).min(second.y + second.height) - first.y.max(second.y);
    width.max(0.0) * height.max(0.0)
}

fn overlap_over_smaller(bounds: (ElementBounds, ElementBounds)) -> Option<f64> {
    let (first, second) = bounds;
    if !valid_bounds(first) || !valid_bounds(second) {
        return None;
    }
    Some(intersection_area(first, second) / area(first).min(area(second)))
}

fn overflow_px(inner: ElementBounds, outer: ElementBounds) -> f64 {
    [
        outer.x - inner.x,
        outer.y - inner.y,
        inner.x + inner.width - (outer.x + outer.width),
        inner.y + inner.height - (outer.y + outer.height),
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

fn normalize_text(text: &str) -> String {
    text.chars()
        .filter(|character| !character.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inspection(elements: Vec<TextElementInspection>) -> ProjectInspection {
        ProjectInspection {
            project: ProjectState {
                kind: "isolated_disposable",
                revision: koharu_scene::Revision::ZERO,
                pages: vec![PageInspection {
                    id: EntityId::new(),
                    label: "page.png".to_owned(),
                    width: 1000.0,
                    height: 1000.0,
                    text_elements: elements,
                    detected_panel_candidates: Vec::new(),
                    logical_dialogue_groups: Vec::new(),
                    render_error: None,
                }],
            },
            configuration: HarnessConfiguration {
                source_language: "en-US".to_owned(),
                target_language: "ko-KR".to_owned(),
                ocr_model: "hayai-ocr".to_owned(),
                required_output_directory: "/tmp".to_owned(),
                required_export_format: "png".to_owned(),
                quality_thresholds: QualityThresholds::default(),
                review_bundle_directory: "/tmp/review".to_owned(),
                external_visual_judge_configured: false,
            },
            repair_history: RepairHistory {
                completed_review_attempts: 0,
                attempted_actions: Vec::new(),
                active_deterministic_plan: None,
                stop_diagnostic: None,
                correction_actions: Vec::new(),
            },
        }
    }

    fn valid_element_at(x: f64, y: f64, translation: &str) -> TextElementInspection {
        let region_id = EntityId::new();
        let source_geometry = ElementGeometry {
            points: vec![
                ElementPoint { x, y },
                ElementPoint { x: x + 100.0, y },
                ElementPoint {
                    x: x + 100.0,
                    y: y + 50.0,
                },
                ElementPoint { x, y: y + 50.0 },
            ],
            bounds: ElementBounds {
                x,
                y,
                width: 100.0,
                height: 50.0,
            },
        };
        TextElementInspection {
            id: EntityId::new(),
            content_id: EntityId::new(),
            source_region_id: Some(region_id),
            source_region_kind: Some("dev.koharu.region.text".to_owned()),
            detected: true,
            required: true,
            text_role: Some("dev.koharu.text.free-text".to_owned()),
            decorative_sfx: None,
            logical_dialogue_memberships: Vec::new(),
            source: Some(SemanticText {
                text: "Hello".to_owned(),
                language: Some("en-US".to_owned()),
            }),
            translation: Some(SemanticText {
                text: translation.to_owned(),
                language: Some("ko-KR".to_owned()),
            }),
            source_writing_mode: None,
            visibility: ElementVisibility {
                local_visible: true,
                local_opacity: 1.0,
                effective_visible: true,
                effective_opacity: 1.0,
            },
            source_geometry: Some(source_geometry.clone()),
            text_safe_region: Some(TextSafeRegion {
                id: region_id,
                kind: "dev.koharu.region.text".to_owned(),
                geometry: source_geometry,
                association: None,
            }),
            verified_ui_panel_anchor: None,
            verified_free_dialogue_anchor: None,
            free_dialogue_anchor_assessment: None,
            typography: None,
            layout_kind: TextLayoutKind::Paragraph,
            authored_layout_geometry: None,
            final_scene: FinalSceneElement {
                eligible: true,
                visible: true,
                opacity: 1.0,
                geometry_visible: true,
                glyph_bounds: Some(ElementBounds {
                    x: x + 10.0,
                    y: y + 10.0,
                    width: 80.0,
                    height: 30.0,
                }),
                layout_bounds: Some(ElementBounds {
                    x,
                    y,
                    width: 100.0,
                    height: 50.0,
                }),
                font_size_px: Some(24.0),
                line_count: Some(1),
                rendered_lines: vec![translation.to_owned()],
                diagnostics: Vec::new(),
                glyph_ink: Some(GlyphInkMask {
                    left: (x + 20.0) as i32,
                    top: (y + 20.0) as i32,
                    width: 1,
                    height: 1,
                    alpha: vec![255],
                }),
            },
        }
    }

    fn difficult_sfx_skip(
        page_id: EntityId,
        element: &TextElementInspection,
    ) -> DecorativeSfxDecision {
        DecorativeSfxDecision {
            schema_version: DECORATIVE_SFX_DECISION_SCHEMA_VERSION,
            disposition: DecorativeSfxDisposition::SkipDifficult,
            review_state: "skipped_difficult_sfx",
            evidence_revision: koharu_scene::Revision::new(3),
            decision_revision: koharu_scene::Revision::new(4),
            page_id,
            original_ordinal: 1,
            element_id: element.id,
            content_id: element.content_id,
            source_region_id: element.source_region_id.unwrap(),
            source_ocr: element.source.clone().unwrap(),
            source_typography: element.typography.clone().unwrap(),
            source_crop_blake3: "crop-digest".to_owned(),
            source_debug_label: "1:ABC123".to_owned(),
            classifier: crate::sfx::SfxClassifier {
                kind: "agent_visual_semantic",
                configured_model: Some("review-model".to_owned()),
                tool_call_id: "call-1".to_owned(),
            },
            evidence: crate::sfx::DecorativeSfxEvidence {
                decorative_visual_form: "irregular impact lettering integrated into the art"
                    .to_owned(),
                sound_effect_page_function: "conveys the impact noise rather than prose".to_owned(),
                legibility_and_translation_value:
                    "overlapping strokes prevent a reliable reading or translation".to_owned(),
                exclusion_of_dialogue_caption_ui_and_general_free_text:
                    "outside containers and not connected to spoken or informational content"
                        .to_owned(),
            },
            confidence: 0.97,
            rationale: "The original crop positively shows difficult decorative impact SFX."
                .to_owned(),
            target_translation_owner: None,
            target_render_owner: None,
        }
    }

    fn narrow_decorative_sfx() -> TextElementInspection {
        let mut element = valid_element_at(20.0, 20.0, "쾅");
        let source_geometry = ElementGeometry {
            points: vec![
                ElementPoint { x: 20.0, y: 20.0 },
                ElementPoint { x: 38.0, y: 20.0 },
                ElementPoint { x: 38.0, y: 31.0 },
                ElementPoint { x: 20.0, y: 31.0 },
            ],
            bounds: ElementBounds {
                x: 20.0,
                y: 20.0,
                width: 18.0,
                height: 11.0,
            },
        };
        element.source = Some(SemanticText {
            text: "BAM".to_owned(),
            language: Some("en-US".to_owned()),
        });
        element.source_geometry = Some(source_geometry.clone());
        element.text_safe_region.as_mut().unwrap().geometry = source_geometry;
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
            writing_mode: None,
            extensions: Default::default(),
        });
        element.final_scene.glyph_bounds = Some(ElementBounds {
            x: 21.0,
            y: 21.0,
            width: 16.0,
            height: 9.0,
        });
        element.final_scene.layout_bounds = Some(ElementBounds {
            x: 20.0,
            y: 20.0,
            width: 18.0,
            height: 11.0,
        });
        element.final_scene.font_size_px = Some(9.0);
        element.final_scene.glyph_ink = Some(GlyphInkMask {
            left: 21,
            top: 21,
            width: 16,
            height: 9,
            alpha: vec![255; 16 * 9],
        });
        element
            .final_scene
            .diagnostics
            .push("text_overflow".to_owned());
        element
    }

    fn translated_sfx_decision(
        page_id: EntityId,
        element: &TextElementInspection,
    ) -> DecorativeSfxDecision {
        let mut decision = difficult_sfx_skip(page_id, element);
        decision.disposition = DecorativeSfxDisposition::Translate;
        decision.review_state = "translated_decorative_sfx";
        decision.evidence.legibility_and_translation_value =
            "the source lettering is legible and carries translatable impact meaning".to_owned();
        decision.rationale =
            "The original crop positively shows legible decorative impact SFX.".to_owned();
        decision.target_translation_owner = Some(element.content_id);
        decision.target_render_owner = Some(element.id);
        decision
    }

    #[test]
    fn translated_decorative_sfx_accepts_a_finite_visible_nine_pixel_raster_in_a_narrow_source_box()
    {
        let mut inspection = inspection(vec![narrow_decorative_sfx()]);
        let page_id = inspection.project.pages[0].id;
        let element = &mut inspection.project.pages[0].text_elements[0];
        element.text_role = Some(DECORATIVE_SFX_ROLE.to_owned());
        element.decorative_sfx = Some(translated_sfx_decision(page_id, element));

        let record = evaluate(&inspection);

        assert!(record.accepted, "{:#?}", record.rejection_reasons);
        assert_eq!(record.pages[0].counts.accepted_elements, 1);
        assert!(record.rejection_reasons.is_empty());
    }

    #[test]
    fn grouping_only_uncontained_translated_decorative_sfx_decision_is_valid() {
        let mut inspection = inspection(vec![narrow_decorative_sfx()]);
        let page_id = inspection.project.pages[0].id;
        let element = &mut inspection.project.pages[0].text_elements[0];
        element.text_role = Some(DECORATIVE_SFX_ROLE.to_owned());
        element.logical_dialogue_memberships = vec![LogicalDialogueMembershipInspection {
            group_id: EntityId::new(),
            primary_render_element_id: element.id,
            target_region_id: EntityId::new(),
            member_ordinal: 1,
        }];
        let source_region = element.source_region_id.unwrap();
        let source_geometry = element.source_geometry.clone().unwrap();
        element.text_safe_region = Some(TextSafeRegion {
            id: source_region,
            kind: "dev.koharu.region.text".to_owned(),
            geometry: source_geometry,
            association: Some(TargetRegionAssociation {
                layout_relation: TargetLayoutRelation::FitsTo,
                source_inside_target_relation: false,
            }),
        });
        assert!(!is_actual_container_bound(element));
        let decision = translated_sfx_decision(page_id, element);
        let page = &inspection.project.pages[0];
        let mut reasons = Vec::new();

        assert!(validate_decorative_sfx_decision(
            page,
            &page.text_elements[0],
            &decision,
            &mut reasons,
        ));
        assert!(reasons.is_empty());
    }

    #[test]
    fn verified_external_container_rejects_translated_decorative_sfx_decision() {
        let mut inspection = inspection(vec![narrow_decorative_sfx()]);
        let page_id = inspection.project.pages[0].id;
        let element = &mut inspection.project.pages[0].text_elements[0];
        element.text_role = Some(DECORATIVE_SFX_ROLE.to_owned());
        let text_safe_region = element.text_safe_region.as_mut().unwrap();
        text_safe_region.id = EntityId::new();
        text_safe_region.association = Some(TargetRegionAssociation {
            layout_relation: TargetLayoutRelation::FlowsIn,
            source_inside_target_relation: true,
        });
        assert!(is_actual_container_bound(element));
        let decision = translated_sfx_decision(page_id, element);
        let page = &inspection.project.pages[0];
        let mut reasons = Vec::new();

        assert!(!validate_decorative_sfx_decision(
            page,
            &page.text_elements[0],
            &decision,
            &mut reasons,
        ));
        assert_eq!(reasons.len(), 1);
        assert_eq!(
            reasons[0].code,
            AcceptanceRejectionCode::InvalidDecorativeSfxDecision
        );
    }

    #[test]
    fn translated_decorative_sfx_accepts_pipeline_normalized_target_typography() {
        let mut inspection = inspection(vec![narrow_decorative_sfx()]);
        inspection.configuration.source_language = "ja-JP".to_owned();
        let page_id = inspection.project.pages[0].id;
        let element = &mut inspection.project.pages[0].text_elements[0];
        element.source = Some(SemanticText {
            text: "ドン".to_owned(),
            language: Some("ja-JP".to_owned()),
        });
        element.source_writing_mode = Some(WritingMode::Vertical);
        element.typography = Some(Typography {
            origin: koharu_scene::Origin::Generated(koharu_scene::Generation {
                producer: koharu_scene::ProducerId::new("dev.koharu.pipeline.detection").unwrap(),
                model: Some("mayocream/koharu-layout-rfdetr-seg-2xl-1152".to_owned()),
                confidence: None,
            }),
            preferred_font: None,
            font_weight: None,
            font_style: None,
            size: None,
            auto_fit: true,
            color: Some([0, 0, 0, 255]),
            stroke_color: None,
            stroke_width: None,
            alignment: None,
            writing_mode: Some(WritingMode::Vertical),
            extensions: Default::default(),
        });
        element.text_role = Some(DECORATIVE_SFX_ROLE.to_owned());
        element.decorative_sfx = Some(translated_sfx_decision(page_id, element));

        element.typography = Some(Typography {
            origin: koharu_scene::Origin::Generated(koharu_scene::Generation {
                producer: koharu_scene::ProducerId::new("dev.koharu.pipeline.detection").unwrap(),
                model: Some("source-region-preprocess-v1".to_owned()),
                confidence: None,
            }),
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

        let record = evaluate(&inspection);

        assert!(record.accepted, "{:#?}", record.rejection_reasons);
        assert_eq!(record.pages[0].counts.accepted_elements, 1);
        assert!(record.rejection_reasons.is_empty());
    }

    #[test]
    fn narrow_nine_pixel_required_text_keeps_legacy_layout_diagnostics_without_rejection() {
        let record = evaluate(&inspection(vec![narrow_decorative_sfx()]));

        assert!(record.accepted, "{:#?}", record.rejection_reasons);
        let measurements = &record.pages[0].elements[0].measurements;
        assert_eq!(measurements.rendered_font_size_px, Some(9.0));
        assert_eq!(measurements.renderer_diagnostics, ["text_overflow"]);
        assert!(
            measurements
                .text_safe_containment
                .as_ref()
                .unwrap()
                .minimum_clearance_px
                .unwrap()
                < record.thresholds.min_text_safe_padding_px
        );
    }

    #[test]
    fn confirmed_difficult_sfx_is_evidence_but_not_a_completeness_or_layout_obligation() {
        let mut inspection = inspection(vec![valid_element_at(20.0, 20.0, "안녕")]);
        let page_id = inspection.project.pages[0].id;
        let element = &mut inspection.project.pages[0].text_elements[0];
        element.required = false;
        element.text_role = Some(SKIPPED_DIFFICULT_SFX_ROLE.to_owned());
        element.translation = None;
        element.visibility.local_visible = false;
        element.visibility.effective_visible = false;
        element.final_scene.visible = false;
        element.typography = Some(Typography {
            origin: koharu_scene::Origin::User,
            preferred_font: None,
            font_weight: None,
            font_style: None,
            size: Some(24.0),
            auto_fit: true,
            color: Some([0, 0, 0, 255]),
            stroke_color: None,
            stroke_width: None,
            alignment: None,
            writing_mode: None,
            extensions: Default::default(),
        });
        element.decorative_sfx = Some(difficult_sfx_skip(page_id, element));

        let record = evaluate(&inspection);

        assert!(record.accepted);
        assert_eq!(record.pages[0].counts.detected_source_elements, 1);
        assert_eq!(record.pages[0].counts.required_source_elements, 0);
        assert_eq!(record.pages[0].counts.skipped_difficult_sfx, 1);
        assert!(record.pages[0].elements.is_empty());
        assert!(record.rejection_reasons.is_empty());
    }

    fn codes(record: &AcceptanceRecord) -> Vec<Value> {
        record
            .rejection_reasons
            .iter()
            .map(|reason| serde_json::to_value(reason.code).unwrap())
            .collect()
    }

    #[test]
    fn prior_english_single_element_behavior_remains_independent_and_accepted() {
        let record = evaluate(&inspection(vec![valid_element_at(
            10.0,
            10.0,
            "안녕하세요",
        )]));
        assert!(record.accepted);
        assert!(record.rejection_reasons.is_empty());
        assert_eq!(record.pages[0].counts.accepted_elements, 1);
        assert!(record.semantic_fidelity_requires_external_review);
    }

    #[test]
    fn finite_visible_required_translation_treats_legacy_placement_thresholds_as_evidence() {
        let mut element = valid_element_at(20.0, 20.0, "안녕");
        element.final_scene.font_size_px = Some(9.0);
        element.final_scene.glyph_bounds = Some(ElementBounds {
            x: 15.0,
            y: 22.0,
            width: 20.0,
            height: 9.0,
        });
        element.final_scene.glyph_ink = Some(GlyphInkMask {
            left: 22,
            top: 22,
            width: 1,
            height: 1,
            alpha: vec![255],
        });

        let accepted = evaluate(&inspection(vec![element.clone()]));

        assert!(accepted.accepted, "{:#?}", accepted.rejection_reasons);
        let measurements = &accepted.pages[0].elements[0].measurements;
        assert!(
            measurements.rendered_font_size_px.unwrap()
                < accepted.thresholds.min_rendered_font_size_px
        );
        assert!(
            measurements.rendered_glyph_height_px.unwrap()
                < accepted.thresholds.min_rendered_glyph_height_px
        );
        assert!(
            measurements
                .target_layout_anchor
                .rendered_area_coverage
                .unwrap()
                < measurements
                    .target_layout_anchor
                    .required_area_coverage
                    .unwrap()
        );
        assert!(
            measurements
                .target_layout_anchor
                .rendered_overflow_px
                .unwrap()
                > accepted.thresholds.max_layout_anchor_overflow_px
        );
        assert!(
            measurements
                .text_safe_containment
                .as_ref()
                .unwrap()
                .minimum_clearance_px
                .unwrap()
                < accepted.thresholds.min_text_safe_padding_px
        );

        let mut hidden = element.clone();
        hidden.visibility.effective_visible = false;
        assert!(!evaluate(&inspection(vec![hidden])).accepted);

        let mut empty = element.clone();
        empty.translation.as_mut().unwrap().text.clear();
        assert!(!evaluate(&inspection(vec![empty])).accepted);

        let mut wrong_language = element.clone();
        wrong_language.translation.as_mut().unwrap().language = Some("en-US".to_owned());
        assert!(!evaluate(&inspection(vec![wrong_language])).accepted);

        element.final_scene.glyph_bounds = Some(ElementBounds {
            x: 995.0,
            y: 22.0,
            width: 20.0,
            height: 9.0,
        });
        assert!(!evaluate(&inspection(vec![element])).accepted);
    }

    fn japanese_vertical_dialogue_with_horizontal_korean() -> TextElementInspection {
        let mut element = valid_element_at(40.0, 20.0, "가로로 배치한 한국어");
        let source_region_id = element.source_region_id.unwrap();
        element.source = Some(SemanticText {
            text: "縦書きの日本語".to_owned(),
            language: Some("ja-JP".to_owned()),
        });
        element.source_writing_mode = Some(WritingMode::Vertical);
        element.source_geometry = Some(ElementGeometry {
            points: vec![
                ElementPoint { x: 40.0, y: 20.0 },
                ElementPoint { x: 88.0, y: 20.0 },
                ElementPoint { x: 88.0, y: 182.0 },
                ElementPoint { x: 40.0, y: 182.0 },
            ],
            bounds: ElementBounds {
                x: 40.0,
                y: 20.0,
                width: 48.0,
                height: 162.0,
            },
        });
        element.typography = Some(Typography {
            origin: koharu_scene::Origin::User,
            preferred_font: None,
            font_weight: None,
            font_style: None,
            size: Some(24.0),
            auto_fit: true,
            color: Some([0, 0, 0, 255]),
            stroke_color: None,
            stroke_width: None,
            alignment: None,
            writing_mode: Some(WritingMode::Horizontal),
            extensions: Default::default(),
        });
        let bubble_id = EntityId::new();
        element.text_safe_region = Some(TextSafeRegion {
            id: bubble_id,
            kind: "dev.koharu.region.bubble".to_owned(),
            geometry: ElementGeometry {
                points: vec![
                    ElementPoint { x: 0.0, y: 0.0 },
                    ElementPoint { x: 133.0, y: 0.0 },
                    ElementPoint { x: 133.0, y: 205.0 },
                    ElementPoint { x: 0.0, y: 205.0 },
                ],
                bounds: ElementBounds {
                    x: 0.0,
                    y: 0.0,
                    width: 133.0,
                    height: 205.0,
                },
            },
            association: Some(TargetRegionAssociation {
                layout_relation: TargetLayoutRelation::FlowsIn,
                source_inside_target_relation: true,
            }),
        });
        element.final_scene.glyph_bounds = Some(ElementBounds {
            x: 20.0,
            y: 80.0,
            width: 90.0,
            height: 40.0,
        });
        element.final_scene.layout_bounds = Some(ElementBounds {
            x: 4.0,
            y: 4.0,
            width: 125.0,
            height: 197.0,
        });
        element.final_scene.glyph_ink = Some(GlyphInkMask {
            left: 50,
            top: 100,
            width: 1,
            height: 1,
            alpha: vec![255],
        });
        assert_ne!(Some(bubble_id), Some(source_region_id));
        element
    }

    fn japanese_to_korean_inspection(element: TextElementInspection) -> ProjectInspection {
        let mut project = inspection(vec![element]);
        project.configuration.source_language = "ja-JP".to_owned();
        project.configuration.target_language = "ko-KR".to_owned();
        let page_id = project.project.pages[0].id;
        if let Some(decision) = project.project.pages[0].text_elements[0]
            .verified_ui_panel_anchor
            .as_mut()
        {
            decision.page_id = page_id;
        }
        if let Some(decision) = project.project.pages[0].text_elements[0]
            .verified_free_dialogue_anchor
            .as_mut()
        {
            decision.page_id = page_id;
        }
        project
    }

    fn verified_adjacent_free_dialogue() -> TextElementInspection {
        let mut element = valid_element_at(60.0, 50.0, "에~");
        let source_region_id = element.source_region_id.unwrap();
        element.text_role = Some(crate::free_dialogue::FREE_DIALOGUE_SOURCE_ROLE.to_owned());
        element.source = Some(SemanticText {
            text: "え〜".to_owned(),
            language: Some("ja-JP".to_owned()),
        });
        element.source_writing_mode = Some(WritingMode::Vertical);
        element.source_geometry = Some(ElementGeometry {
            points: vec![
                ElementPoint { x: 60.0, y: 50.0 },
                ElementPoint { x: 76.0, y: 50.0 },
                ElementPoint { x: 76.0, y: 90.0 },
                ElementPoint { x: 60.0, y: 90.0 },
            ],
            bounds: ElementBounds {
                x: 60.0,
                y: 50.0,
                width: 16.0,
                height: 40.0,
            },
        });
        element.typography = Some(Typography {
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
        });
        let raster = image::GrayImage::from_pixel(240, 180, image::Luma([244]));
        let assessment = crate::free_dialogue::assess_free_dialogue_anchor(
            &raster,
            crate::free_dialogue::FreeDialogueAnchorInput {
                source_region_id,
                source_bounds: element.source_geometry.as_ref().unwrap().bounds,
                source_text: "え〜",
                target_text: "에~",
                source_scene_role: crate::free_dialogue::FREE_DIALOGUE_SOURCE_ROLE,
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
        let selected = assessment.selected_candidate.clone().unwrap();
        let target_region_id = EntityId::new();
        element.text_safe_region = Some(TextSafeRegion {
            id: target_region_id,
            kind: koharu_scene::TextRegion::KIND.to_owned(),
            geometry: ElementGeometry {
                points: vec![
                    ElementPoint {
                        x: selected.bounds.x,
                        y: selected.bounds.y,
                    },
                    ElementPoint {
                        x: selected.bounds.x + selected.bounds.width,
                        y: selected.bounds.y,
                    },
                    ElementPoint {
                        x: selected.bounds.x + selected.bounds.width,
                        y: selected.bounds.y + selected.bounds.height,
                    },
                    ElementPoint {
                        x: selected.bounds.x,
                        y: selected.bounds.y + selected.bounds.height,
                    },
                ],
                bounds: selected.bounds,
            },
            association: Some(TargetRegionAssociation {
                layout_relation: TargetLayoutRelation::FitsTo,
                source_inside_target_relation: false,
            }),
        });
        let inner = inset_bounds(selected.bounds, 4.0).unwrap();
        element.final_scene.glyph_bounds = Some(ElementBounds {
            x: inner.x + 1.0,
            y: inner.y + 1.0,
            width: inner.width - 2.0,
            height: 10.0,
        });
        element.final_scene.layout_bounds = Some(inner);
        element.final_scene.font_size_px = Some(12.0);
        element.final_scene.line_count = Some(1);
        element.final_scene.glyph_ink = Some(GlyphInkMask {
            left: (inner.x + 2.0).round() as i32,
            top: (inner.y + 2.0).round() as i32,
            width: 1,
            height: 1,
            alpha: vec![255],
        });
        element.verified_free_dialogue_anchor = Some(FreeDialogueAnchorDecision {
            schema_version: crate::free_dialogue::FREE_DIALOGUE_ANCHOR_SCHEMA_VERSION,
            decision: "source_raster_verified_adjacent_free_dialogue_anchor",
            evidence_revision: koharu_scene::Revision::new(4),
            decision_revision: koharu_scene::Revision::new(5),
            page_id: EntityId::new(),
            element_id: element.id,
            content_id: element.content_id,
            source_region_id,
            target_region_id,
            source_ocr: element.source.clone().unwrap(),
            target_translation: element.translation.clone().unwrap(),
            source_bounds: element.source_geometry.as_ref().unwrap().bounds,
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
            detector_producer: crate::free_dialogue::RASTER_FREE_DIALOGUE_DETECTOR,
            detector_version: crate::free_dialogue::RASTER_FREE_DIALOGUE_DETECTOR_VERSION,
            detector_input: "original_source_pixels",
            rejected_candidates: assessment
                .candidates
                .iter()
                .filter(|candidate| !candidate.accepted)
                .cloned()
                .collect(),
        });
        element.free_dialogue_anchor_assessment = Some(assessment);
        element
    }

    #[test]
    fn source_raster_verified_free_dialogue_anchor_passes_every_normal_layout_gate() {
        let element = verified_adjacent_free_dialogue();
        let record = evaluate(&japanese_to_korean_inspection(element));

        assert!(record.accepted, "{:?}", codes(&record));
        let measurements = &record.pages[0].elements[0].measurements;
        assert_eq!(
            measurements.target_layout_anchor.kind,
            LayoutAnchorKind::AdjacentFreeDialogueTextSafeAnchor
        );
        assert!(measurements.free_dialogue_anchor_evidence.is_some());
        assert!(
            measurements
                .text_safe_containment
                .as_ref()
                .unwrap()
                .minimum_clearance_px
                .unwrap()
                >= 4.0
        );
        assert!(measurements.rendered_font_size_px.unwrap() >= 12.0);
        assert!(measurements.rendered_glyph_height_px.unwrap() >= 10.0);
    }

    fn verified_ui_panel_label() -> TextElementInspection {
        let mut element = valid_element_at(50.0, 25.0, "고장 중");
        let source_region_id = element.source_region_id.unwrap();
        element.source = Some(SemanticText {
            text: "故障中".to_owned(),
            language: Some("ja-JP".to_owned()),
        });
        element.translation = Some(SemanticText {
            text: "고장 중".to_owned(),
            language: Some("ko-KR".to_owned()),
        });
        element.source_writing_mode = Some(WritingMode::Horizontal);
        let source_geometry = ElementGeometry {
            points: vec![
                ElementPoint { x: 50.0, y: 25.0 },
                ElementPoint {
                    x: 118.815,
                    y: 25.0,
                },
                ElementPoint {
                    x: 118.815,
                    y: 43.75,
                },
                ElementPoint { x: 50.0, y: 43.75 },
            ],
            bounds: ElementBounds {
                x: 50.0,
                y: 25.0,
                width: 68.815,
                height: 18.75,
            },
        };
        element.source_geometry = Some(source_geometry);
        element.text_role = Some(UI_TEXT_ROLE.to_owned());
        element.typography = Some(Typography {
            origin: koharu_scene::Origin::User,
            preferred_font: None,
            font_weight: None,
            font_style: None,
            size: Some(12.0),
            auto_fit: true,
            color: Some([0, 0, 0, 255]),
            stroke_color: None,
            stroke_width: None,
            alignment: Some(koharu_scene::TextAlignment::Center),
            writing_mode: Some(WritingMode::Horizontal),
            extensions: Default::default(),
        });
        let panel_id = EntityId::new();
        let panel_geometry = ElementGeometry {
            points: vec![
                ElementPoint { x: 10.0, y: 10.0 },
                ElementPoint { x: 160.0, y: 10.0 },
                ElementPoint { x: 160.0, y: 60.0 },
                ElementPoint { x: 10.0, y: 60.0 },
            ],
            bounds: ElementBounds {
                x: 10.0,
                y: 10.0,
                width: 150.0,
                height: 50.0,
            },
        };
        let panel = DetectedPanelCandidate {
            region_id: panel_id,
            region_kind: koharu_scene::PanelRegion::KIND.to_owned(),
            geometry: panel_geometry.clone(),
            detection_label: "panel".to_owned(),
            detection_confidence: 0.96,
            detector_producer: crate::ui_panel::RASTER_PANEL_DETECTOR.to_owned(),
            detector_model: Some(crate::ui_panel::RASTER_PANEL_DETECTOR_VERSION.to_owned()),
            raster_evidence: crate::ui_panel::test_raster_panel_evidence(
                source_region_id,
                element.source_geometry.as_ref().unwrap().bounds,
                panel_geometry.bounds,
            ),
        };
        element.text_safe_region = Some(TextSafeRegion {
            id: panel_id,
            kind: koharu_scene::PanelRegion::KIND.to_owned(),
            geometry: panel_geometry,
            association: Some(TargetRegionAssociation {
                layout_relation: TargetLayoutRelation::FlowsIn,
                source_inside_target_relation: true,
            }),
        });
        element.verified_ui_panel_anchor = Some(UiPanelAnchorDecision {
            schema_version: 2,
            decision: "verified_required_ui_panel_text_anchor",
            evidence_revision: koharu_scene::Revision::new(3),
            decision_revision: koharu_scene::Revision::new(4),
            page_id: EntityId::new(),
            original_ordinal: 10,
            element_id: element.id,
            content_id: element.content_id,
            source_region_id,
            source_ocr: element.source.clone().unwrap(),
            source_crop_blake3: "crop".to_owned(),
            source_debug_label: "10:AF0F7E".to_owned(),
            panel,
            source_containment_ratio: 1.0,
            source_intersection_ratio: 0.172,
            classifier: crate::ui_panel::UiPanelClassifier {
                kind: "agent_visual_semantic",
                configured_model: Some("review-model".to_owned()),
                tool_call_id: "call-ui".to_owned(),
            },
            evidence: crate::ui_panel::UiPanelEvidence {
                ui_role_and_function: "device status label".to_owned(),
                finite_visible_panel_or_screen: "finite rectangular device display".to_owned(),
                source_to_panel_relation: "source label is contained by the display".to_owned(),
                text_safe_interior: "display interior is visibly clear for status text".to_owned(),
                visual_or_detection_provenance: "detected panel confirmed in original pixels"
                    .to_owned(),
            },
            confidence: 0.97,
            association_reason: "verified_required_ui_text_inside_explicit_detected_panel"
                .to_owned(),
        });
        element.final_scene.glyph_bounds = Some(ElementBounds {
            x: 65.0,
            y: 29.0,
            width: 40.0,
            height: 12.0,
        });
        element.final_scene.layout_bounds = Some(ElementBounds {
            x: 14.0,
            y: 14.0,
            width: 142.0,
            height: 42.0,
        });
        element.final_scene.font_size_px = Some(12.0);
        element.final_scene.line_count = Some(1);
        element.final_scene.glyph_ink = Some(GlyphInkMask {
            left: 75,
            top: 29,
            width: 1,
            height: 12,
            alpha: vec![255; 12],
        });
        element
    }

    #[test]
    fn small_source_ui_label_accepts_only_against_verified_panel_text_safe_interior() {
        let element = verified_ui_panel_label();
        let panel = element
            .verified_ui_panel_anchor
            .as_ref()
            .unwrap()
            .panel
            .clone();
        let mut inspection = japanese_to_korean_inspection(element);
        inspection.project.pages[0].detected_panel_candidates = vec![panel];
        let record = evaluate(&inspection);

        assert!(record.accepted, "{:?}", codes(&record));
        let measurements = &record.pages[0].elements[0].measurements;
        assert_eq!(
            measurements.target_layout_anchor.kind,
            LayoutAnchorKind::UiPanelTextSafeInterior
        );
        assert_eq!(measurements.rendered_font_size_px, Some(12.0));
        assert!(
            measurements
                .text_safe_containment
                .as_ref()
                .unwrap()
                .minimum_clearance_px
                .unwrap()
                >= 4.0
        );
    }

    #[test]
    fn weak_or_disallowed_ui_panel_claims_remain_source_bound() {
        for mode in ["unverified", "weak", "dialogue", "free_text"] {
            let mut element = verified_ui_panel_label();
            match mode {
                "unverified" => {
                    element.verified_ui_panel_anchor = None;
                    let source_geometry = element.source_geometry.clone().unwrap();
                    element.text_safe_region = Some(TextSafeRegion {
                        id: element.source_region_id.unwrap(),
                        kind: "dev.koharu.region.text".to_owned(),
                        geometry: source_geometry,
                        association: None,
                    });
                }
                "weak" => {
                    element
                        .verified_ui_panel_anchor
                        .as_mut()
                        .unwrap()
                        .source_containment_ratio = 0.89
                }
                "dialogue" => element.text_role = Some("dev.koharu.text.dialogue".to_owned()),
                "free_text" => element.text_role = Some("dev.koharu.text.free-text".to_owned()),
                _ => unreachable!(),
            }
            let record = evaluate(&japanese_to_korean_inspection(element));
            assert_eq!(
                record.pages[0].elements[0]
                    .measurements
                    .target_layout_anchor
                    .kind,
                LayoutAnchorKind::SourceRegion,
                "mode {mode}"
            );
            assert!(
                record.accepted,
                "mode {mode}: {:#?}",
                record.rejection_reasons
            );
        }
    }

    fn grouped_japanese_dialogue() -> ProjectInspection {
        let mut primary = japanese_vertical_dialogue_with_horizontal_korean();
        let mut secondary = japanese_vertical_dialogue_with_horizontal_korean();
        primary.source.as_mut().unwrap().text = "右の断片".to_owned();
        secondary.source.as_mut().unwrap().text = "左の断片".to_owned();
        secondary.source_geometry.as_mut().unwrap().bounds.x = 20.0;
        secondary.text_safe_region = primary.text_safe_region.clone();
        secondary.translation = None;
        secondary.visibility.local_visible = false;
        secondary.visibility.effective_visible = false;
        secondary.final_scene.visible = false;
        secondary.final_scene.geometry_visible = false;
        let group_id = primary.content_id;
        let target_region_id = primary.text_safe_region.as_ref().unwrap().id;
        let members = vec![
            LogicalDialogueMemberInspection {
                ordinal: 1,
                element_id: primary.id,
                content_id: primary.content_id,
                source_region_id: primary.source_region_id.unwrap(),
                source_text: primary.source.as_ref().unwrap().text.clone(),
            },
            LogicalDialogueMemberInspection {
                ordinal: 2,
                element_id: secondary.id,
                content_id: secondary.content_id,
                source_region_id: secondary.source_region_id.unwrap(),
                source_text: secondary.source.as_ref().unwrap().text.clone(),
            },
        ];
        let membership = |ordinal| LogicalDialogueMembershipInspection {
            group_id,
            primary_render_element_id: primary.id,
            target_region_id,
            member_ordinal: ordinal,
        };
        primary.logical_dialogue_memberships = vec![membership(1)];
        secondary.logical_dialogue_memberships = vec![membership(2)];
        let mut inspection = japanese_to_korean_inspection(primary.clone());
        inspection.project.pages[0].text_elements = vec![primary, secondary];
        inspection.project.pages[0].logical_dialogue_groups =
            vec![LogicalDialogueGroupInspection {
                group_id,
                primary_render_element_id: members[0].element_id,
                target_region_id,
                logical_source_text: members
                    .iter()
                    .map(|member| member.source_text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
                members,
            }];
        inspection
    }

    #[test]
    fn logical_dialogue_measures_one_render_owner_and_preserves_all_source_members() {
        let inspection = grouped_japanese_dialogue();
        let record = evaluate(&inspection);

        assert!(record.accepted, "{:?}", codes(&record));
        assert_eq!(record.pages[0].counts.required_source_elements, 2);
        assert_eq!(record.pages[0].counts.translation_present, 1);
        assert_eq!(record.pages[0].counts.visible_translations, 1);
        assert_eq!(record.pages[0].counts.accepted_elements, 2);
        assert!(record.pages[0].elements[0].evaluated_render_owner);
        assert!(!record.pages[0].elements[1].evaluated_render_owner);
        assert_eq!(
            record.pages[0].elements[0]
                .measurements
                .target_layout_anchor
                .kind,
            LayoutAnchorKind::TextSafeRegion
        );
        assert!(
            record.pages[0].elements[0]
                .measurements
                .rendered_font_size_px
                .unwrap()
                >= record.thresholds.min_rendered_font_size_px
        );
        assert!(
            record.pages[0].elements[0]
                .measurements
                .target_layout_anchor
                .rendered_area_coverage
                .unwrap()
                >= record.thresholds.min_layout_anchor_area_coverage
        );
        assert!(record.pages[0].pairs.is_empty());
    }

    #[test]
    fn source_member_without_translation_or_exactly_one_group_owner_is_rejected() {
        let mut inspection = grouped_japanese_dialogue();
        inspection.project.pages[0].text_elements[1]
            .logical_dialogue_memberships
            .clear();
        let record = evaluate(&inspection);

        assert!(!record.accepted);
        assert!(codes(&record).contains(&Value::String("translation_missing".to_owned())));
        assert!(codes(&record).contains(&Value::String(
            "logical_dialogue_member_mismatch".to_owned()
        )));
    }

    #[test]
    fn logical_dialogue_below_legacy_font_floor_does_not_enter_repair_plan() {
        let mut inspection = grouped_japanese_dialogue();
        inspection.project.pages[0].text_elements[0]
            .final_scene
            .font_size_px = Some(10.57);
        let record = evaluate(&inspection);
        let plan = crate::repair::deterministic_repair_plan(
            &record.rejection_reasons,
            koharu_scene::Revision::new(18),
            Some(&inspection),
        );
        assert!(record.accepted, "{:#?}", record.rejection_reasons);
        assert!(plan.blocking_failures.is_empty());
        assert_eq!(plan.unresolved_failure_count, 0);
    }

    #[test]
    fn vertical_japanese_uses_verified_inset_bubble_anchor_for_horizontal_korean() {
        let element = japanese_vertical_dialogue_with_horizontal_korean();
        let record = evaluate(&japanese_to_korean_inspection(element));

        assert!(record.accepted, "{:?}", codes(&record));
        let measurements = &record.pages[0].elements[0].measurements;
        assert_eq!(
            measurements.target_layout_anchor.kind,
            LayoutAnchorKind::TextSafeRegion
        );
        assert_eq!(
            measurements.target_layout_anchor.bounds,
            Some(ElementBounds {
                x: 4.0,
                y: 4.0,
                width: 125.0,
                height: 197.0,
            })
        );
        assert_eq!(
            measurements.target_layout_anchor.source_writing_mode,
            Some(WritingMode::Vertical)
        );
        assert_eq!(
            measurements.target_layout_anchor.target_writing_mode,
            Some(WritingMode::Horizontal)
        );
        assert_eq!(
            measurements.target_layout_anchor.association_confidence,
            Some(1.0)
        );
        assert!(measurements.source_region_overflow_px.unwrap() > 2.0);
        assert_eq!(
            measurements.target_layout_anchor.rendered_overflow_px,
            Some(0.0)
        );
    }

    #[test]
    fn short_korean_label_uses_expected_ink_occupancy_only_in_verified_target_anchor() {
        let mut element = japanese_vertical_dialogue_with_horizontal_korean();
        element.translation.as_mut().unwrap().text = "고장 중".to_owned();
        element.final_scene.font_size_px = Some(14.0);
        element.final_scene.line_count = Some(1);
        element.final_scene.glyph_bounds = Some(ElementBounds {
            x: 50.0,
            y: 95.0,
            width: 30.0,
            height: 14.0,
        });
        element.final_scene.glyph_ink = Some(GlyphInkMask {
            left: 60,
            top: 100,
            width: 1,
            height: 1,
            alpha: vec![255],
        });

        let inspection = japanese_to_korean_inspection(element.clone());
        let record = evaluate(&inspection);
        assert!(record.accepted, "{:?}", codes(&record));
        let anchor = &record.pages[0].elements[0]
            .measurements
            .target_layout_anchor;
        assert_eq!(
            anchor.coverage_policy,
            LayoutAnchorCoveragePolicy::VerifiedTargetExpectedInk
        );
        assert_eq!(anchor.expected_text_units, 3);
        assert_eq!(anchor.target_anchor_area_px2, Some(125.0 * 197.0));
        assert_eq!(anchor.measured_ink_bounds_area_px2, Some(30.0 * 14.0));
        assert_eq!(anchor.required_area_coverage, Some(0.01));
        assert!(
            anchor.rendered_area_coverage.unwrap()
                < record.thresholds.min_layout_anchor_area_coverage
        );

        let mut outside = element;
        outside.final_scene.glyph_ink = Some(GlyphInkMask {
            left: 2,
            top: 2,
            width: 1,
            height: 1,
            alpha: vec![255],
        });
        let outside_record = evaluate(&japanese_to_korean_inspection(outside));
        assert!(
            outside_record.accepted,
            "{:#?}",
            outside_record.rejection_reasons
        );
        assert!(
            outside_record.pages[0].elements[0]
                .measurements
                .text_safe_containment
                .as_ref()
                .unwrap()
                .minimum_clearance_px
                .unwrap()
                < outside_record.thresholds.min_text_safe_padding_px
        );
    }

    #[test]
    fn short_korean_label_below_font_and_glyph_minima_remains_measured() {
        let mut element = japanese_vertical_dialogue_with_horizontal_korean();
        element.translation.as_mut().unwrap().text = "고장 중".to_owned();
        element.final_scene.font_size_px = Some(9.0);
        element.final_scene.line_count = Some(1);
        element.final_scene.glyph_bounds = Some(ElementBounds {
            x: 50.0,
            y: 95.0,
            width: 34.0,
            height: 9.0,
        });
        let record = evaluate(&japanese_to_korean_inspection(element));
        assert!(record.accepted, "{:#?}", record.rejection_reasons);
        let measurements = &record.pages[0].elements[0].measurements;
        assert_eq!(measurements.rendered_font_size_px, Some(9.0));
        assert_eq!(measurements.rendered_glyph_height_px, Some(9.0));
    }

    #[test]
    fn long_korean_dialogue_keeps_configured_coverage_floor_in_verified_anchor() {
        let mut element = japanese_vertical_dialogue_with_horizontal_korean();
        element.translation.as_mut().unwrap().text =
            "이 대사는 같은 말풍선 안에서 충분히 길어서 작은 글자 밀도를 허용하면 안 되고 읽기 좋은 크기를 유지해야 한다"
                .to_owned();
        element.final_scene.font_size_px = Some(14.0);
        element.final_scene.line_count = Some(6);
        element.final_scene.glyph_bounds = Some(ElementBounds {
            x: 50.0,
            y: 95.0,
            width: 30.0,
            height: 14.0,
        });
        let record = evaluate(&japanese_to_korean_inspection(element));
        let anchor = &record.pages[0].elements[0]
            .measurements
            .target_layout_anchor;

        assert_eq!(
            anchor.coverage_policy,
            LayoutAnchorCoveragePolicy::ConfiguredLongDialogueFloor
        );
        assert!(
            anchor.expected_ink_area_px2 / anchor.target_anchor_area_px2.unwrap()
                >= record.thresholds.min_layout_anchor_area_coverage
        );
        assert_eq!(
            anchor.required_area_coverage,
            Some(record.thresholds.min_layout_anchor_area_coverage)
        );
        assert!(record.accepted, "{:#?}", record.rejection_reasons);
        assert!(anchor.rendered_area_coverage.unwrap() < anchor.required_area_coverage.unwrap());
    }

    #[test]
    fn vertical_japanese_horizontal_korean_outside_inset_bubble_is_recorded_as_overflow() {
        let mut element = japanese_vertical_dialogue_with_horizontal_korean();
        element.final_scene.glyph_bounds = Some(ElementBounds {
            x: 135.0,
            y: 80.0,
            width: 20.0,
            height: 40.0,
        });
        element.final_scene.glyph_ink = Some(GlyphInkMask {
            left: 140,
            top: 100,
            width: 1,
            height: 1,
            alpha: vec![255],
        });
        let record = evaluate(&japanese_to_korean_inspection(element));

        assert!(record.accepted, "{:#?}", record.rejection_reasons);
        let anchor = &record.pages[0].elements[0]
            .measurements
            .target_layout_anchor;
        assert_eq!(anchor.kind, LayoutAnchorKind::TextSafeRegion);
        assert!(
            anchor.rendered_overflow_px.unwrap() > record.thresholds.max_layout_anchor_overflow_px
        );
    }

    #[test]
    fn vertical_japanese_source_self_fits_to_stays_source_bound() {
        let mut element = japanese_vertical_dialogue_with_horizontal_korean();
        let source_geometry = element.source_geometry.clone().unwrap();
        element.text_safe_region = Some(TextSafeRegion {
            id: element.source_region_id.unwrap(),
            kind: "dev.koharu.region.text".to_owned(),
            geometry: source_geometry,
            association: Some(TargetRegionAssociation {
                layout_relation: TargetLayoutRelation::FitsTo,
                source_inside_target_relation: false,
            }),
        });
        assert!(!is_actual_container_bound(&element));
        let record = evaluate(&japanese_to_korean_inspection(element));

        assert!(record.accepted, "{:#?}", record.rejection_reasons);
        assert_eq!(
            record.pages[0].elements[0]
                .measurements
                .target_layout_anchor
                .kind,
            LayoutAnchorKind::SourceRegion
        );
    }

    #[test]
    fn rejects_requested_source_language_mismatch() {
        let mut element = valid_element_at(10.0, 10.0, "안녕하세요");
        element.source.as_mut().unwrap().language = Some("ja-JP".to_owned());
        let record = evaluate(&inspection(vec![element]));
        assert!(!record.accepted);
        assert!(codes(&record).contains(&Value::String("source_language_mismatch".to_owned())));
        assert_eq!(
            record.pages[0].elements[0]
                .measurements
                .requested_source_language,
            "en-US"
        );
        assert_eq!(
            record.pages[0].elements[0]
                .measurements
                .actual_source_language
                .as_deref(),
            Some("ja-JP")
        );
    }

    #[test]
    fn records_tiny_rendering_and_low_source_region_coverage() {
        let mut element = valid_element_at(0.0, 0.0, "작은 글자");
        element.source_geometry.as_mut().unwrap().bounds = ElementBounds {
            x: 0.0,
            y: 0.0,
            width: 692.0,
            height: 336.0,
        };
        element.final_scene.glyph_bounds = Some(ElementBounds {
            x: 100.0,
            y: 100.0,
            width: 424.0,
            height: 24.0,
        });
        element.final_scene.font_size_px = Some(9.0);
        let record = evaluate(&inspection(vec![element]));
        assert!(record.accepted, "{:#?}", record.rejection_reasons);
        let coverage = record.pages[0].elements[0]
            .measurements
            .source_region_area_coverage
            .unwrap();
        assert!((coverage - (424.0 * 24.0 / (692.0 * 336.0))).abs() < 1e-9);
        let anchor = &record.pages[0].elements[0]
            .measurements
            .target_layout_anchor;
        assert_eq!(
            anchor.coverage_policy,
            LayoutAnchorCoveragePolicy::ConfiguredSourceAnchorFloor
        );
        assert_eq!(
            anchor.required_area_coverage,
            Some(record.thresholds.min_layout_anchor_area_coverage)
        );
    }

    #[test]
    fn rejects_page_clipping_but_only_records_source_anchor_overflow() {
        let mut element = valid_element_at(920.0, 920.0, "잘림");
        element.final_scene.glyph_bounds = Some(ElementBounds {
            x: 930.0,
            y: 930.0,
            width: 90.0,
            height: 80.0,
        });
        let record = evaluate(&inspection(vec![element]));
        let codes = codes(&record);
        assert!(codes.contains(&Value::String("rendered_text_outside_page".to_owned())));
        assert!(!codes.contains(&Value::String(
            "rendered_text_outside_source_region".to_owned()
        )));
    }

    #[test]
    fn records_glyph_crossing_balloon_interior_without_rejection() {
        let mut element = valid_element_at(0.0, 0.0, "경계 침범");
        element.source_geometry.as_mut().unwrap().bounds = ElementBounds {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
        };
        element.final_scene.glyph_bounds = Some(ElementBounds {
            x: 8.0,
            y: 45.0,
            width: 20.0,
            height: 10.0,
        });
        element.final_scene.layout_bounds = Some(ElementBounds {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
        });
        element.final_scene.glyph_ink = Some(GlyphInkMask {
            left: 10,
            top: 49,
            width: 1,
            height: 1,
            alpha: vec![255],
        });
        element.text_safe_region = Some(TextSafeRegion {
            id: EntityId::new(),
            kind: "dev.koharu.region.bubble".to_owned(),
            geometry: ElementGeometry {
                // The inward notch models a balloon wall that a rectangular source box misses.
                points: vec![
                    ElementPoint { x: 0.0, y: 0.0 },
                    ElementPoint { x: 100.0, y: 0.0 },
                    ElementPoint { x: 100.0, y: 100.0 },
                    ElementPoint { x: 0.0, y: 100.0 },
                    ElementPoint { x: 30.0, y: 50.0 },
                ],
                bounds: ElementBounds {
                    x: 0.0,
                    y: 0.0,
                    width: 100.0,
                    height: 100.0,
                },
            },
            association: None,
        });

        let record = evaluate(&inspection(vec![element]));
        assert!(record.accepted, "{:#?}", record.rejection_reasons);
        assert!(!codes(&record).contains(&Value::String(
            "rendered_text_outside_source_region".to_owned()
        )));
        let containment = record.pages[0].elements[0]
            .measurements
            .text_safe_containment
            .as_ref()
            .unwrap();
        assert!(containment.minimum_clearance_px.unwrap() < 0.0);
        assert_eq!(containment.violation_pixels, 1);
        assert_eq!(containment.violations.len(), 1);
    }

    #[test]
    fn text_safe_padding_threshold_changes_evidence_without_changing_acceptance() {
        let mut element = valid_element_at(0.0, 0.0, "안쪽 여백");
        element.final_scene.glyph_bounds = Some(ElementBounds {
            x: 2.0,
            y: 10.0,
            width: 80.0,
            height: 30.0,
        });
        element.final_scene.glyph_ink = Some(GlyphInkMask {
            left: 2,
            top: 10,
            width: 80,
            height: 30,
            alpha: vec![255; 80 * 30],
        });
        let mut inspection = inspection(vec![element]);

        let default_record = evaluate(&inspection);
        assert!(
            default_record.accepted,
            "{:#?}",
            default_record.rejection_reasons
        );
        let clearance = default_record.pages[0].elements[0]
            .measurements
            .text_safe_containment
            .as_ref()
            .unwrap()
            .minimum_clearance_px
            .unwrap();
        assert!(clearance > 1.0 && clearance < 4.0);

        inspection
            .configuration
            .quality_thresholds
            .min_text_safe_padding_px = 1.0;
        let relaxed_padding_record = evaluate(&inspection);
        assert!(relaxed_padding_record.accepted);
        assert_eq!(
            relaxed_padding_record.pages[0].elements[0]
                .measurements
                .text_safe_containment
                .as_ref()
                .unwrap()
                .violation_pixels,
            0
        );
    }

    #[test]
    fn rejects_overlapping_duplicate_translated_regions() {
        let first = valid_element_at(100.0, 100.0, "중복 대사");
        let second = valid_element_at(105.0, 105.0, " 중복대사 ");
        let record = evaluate(&inspection(vec![first, second]));
        let codes = codes(&record);
        assert!(codes.contains(&Value::String("translated_region_overlap".to_owned())));
        assert!(codes.contains(&Value::String("duplicate_translation_region".to_owned())));
        assert!(codes.contains(&Value::String("source_region_overlap".to_owned())));
        assert_eq!(record.pages[0].pairs.len(), 1);
        assert!(!record.pages[0].pairs[0].accepted);
    }

    #[test]
    fn rejects_unsupported_nonfinite_layout() {
        let mut element = valid_element_at(10.0, 10.0, "오류");
        element.final_scene.font_size_px = Some(f64::NAN);
        element
            .final_scene
            .diagnostics
            .push("text_overflow".to_owned());
        let record = evaluate(&inspection(vec![element]));
        assert!(codes(&record).contains(&Value::String("unsupported_layout".to_owned())));
        assert_eq!(
            record.pages[0].elements[0].measurements.invalid_metrics,
            ["rendered_font_size_non_finite"]
        );
        serde_json::to_value(record).unwrap();
    }

    #[test]
    fn validates_configurable_threshold_ranges() {
        let invalid = QualityThresholds {
            min_layout_anchor_area_coverage: 1.1,
            ..QualityThresholds::default()
        };
        assert!(invalid.validate().is_err());
        let negative_padding = QualityThresholds {
            min_text_safe_padding_px: -0.1,
            ..QualityThresholds::default()
        };
        assert!(negative_padding.validate().is_err());
    }
}
