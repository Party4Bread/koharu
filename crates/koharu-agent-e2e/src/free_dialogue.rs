use image::GrayImage;
use koharu_scene::{EntityId, Revision, WritingMode};
use serde::Serialize;
use std::collections::BTreeMap;
use unicode_segmentation::UnicodeSegmentation as _;

use crate::acceptance::{ElementBounds, SemanticText};

pub(crate) const DIALOGUE_ROLE: &str = "dev.koharu.text.dialogue";
pub(crate) const FREE_DIALOGUE_SOURCE_ROLE: &str = "dev.koharu.text.free-text";
pub(crate) const FREE_DIALOGUE_ANCHOR_SCHEMA_VERSION: u32 = 3;
pub(crate) const ADJACENT_FREE_DIALOGUE_DETECTION_KIND: &str =
    "dev.koharu.region.adjacent-free-dialogue-anchor";
pub(crate) const RASTER_FREE_DIALOGUE_DETECTOR: &str =
    "dev.koharu.agent-e2e.source-raster-free-dialogue-anchor";
pub(crate) const RASTER_FREE_DIALOGUE_DETECTOR_VERSION: &str = "adjacent-text-safe-v4";
pub(crate) const MINIMUM_FREE_DIALOGUE_ANCHOR_SCORE: f64 = 0.82;

const MINIMUM_FONT_PX: f64 = 12.0;
const MINIMUM_GLYPH_HEIGHT_PX: f64 = 10.0;
const REQUIRED_CLEARANCE_PX: f64 = 4.0;
const MAX_EDGE_DENSITY: f64 = 0.075;
const MAX_DARK_INK_RATIO: f64 = 0.055;
const MAX_LUMA_STANDARD_DEVIATION: f64 = 18.0;
const MIN_TARGET_INK_CONTRAST: f64 = 104.0;
const MAX_BOUNDARY_EDGE_DENSITY: f64 = 0.12;
const MAX_BOUNDARY_AXIS_COVERAGE: f64 = 0.62;
const EDGE_DELTA: i16 = 32;
const DARK_LUMA: u8 = 112;
const MAXIMUM_SOURCE_BOUND_FALLBACK_TARGET_UNITS: usize = 2;
pub(crate) const NO_ADJACENT_CANDIDATE_PASSED_EVIDENCE: &str =
    "no_adjacent_candidate_passed_every_raster_gate";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AdjacentDirection {
    Right,
    Left,
    Above,
    Below,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FreeDialogueRoleGate {
    pub source_scene_role: String,
    pub classified_role: &'static str,
    pub exact_gate: &'static str,
    pub source_text: String,
    pub required: bool,
    pub no_finite_verified_container_relation: bool,
    pub source_bounds: ElementBounds,
    pub compact_vertical_source_geometry: bool,
    pub eligible_uncontained_source: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FreeDialogueWritingModeGate {
    pub source_language: String,
    pub target_language: String,
    pub source_writing_mode: Option<WritingMode>,
    pub target_writing_mode: Option<WritingMode>,
    pub exact_gate: &'static str,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FreeDialogueRoomEvidence {
    pub target_grapheme_units: usize,
    pub minimum_font_px: f64,
    pub minimum_glyph_height_px: f64,
    pub all_edge_clearance_px: f64,
    pub required_outer_width_px: f64,
    pub required_outer_height_px: f64,
    pub usable_width_px: f64,
    pub usable_height_px: f64,
    pub fits_target_length: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FreeDialoguePixelEvidence {
    pub sampled_pixels: u32,
    pub mean_luma: f64,
    pub luma_standard_deviation: f64,
    pub minimum_luma: u8,
    pub maximum_luma: u8,
    pub dark_ink_ratio: f64,
    pub edge_density: f64,
    pub boundary_sampled_pixels: u32,
    pub boundary_edge_density: f64,
    pub boundary_max_axis_edge_coverage: f64,
    pub overlaps_existing_text_bounds: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FreeDialogueContrastEvidence {
    pub target_ink_luma: u8,
    pub background_mean_luma: f64,
    pub absolute_contrast: f64,
    pub minimum_contrast: f64,
    pub contrastable: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FreeDialogueAssociationEvidence {
    pub source_center_distance_px: f64,
    pub source_edge_distance_px: f64,
    pub maximum_source_edge_distance_px: f64,
    pub source_edge_distance_within_cap: bool,
    pub nearest_other_source_center_distance_px: Option<f64>,
    pub attribution_margin_px: Option<f64>,
    pub has_required_attribution_margin: bool,
    pub reading_order_preserved: bool,
    pub preserves_reading_order_and_attribution: bool,
    pub confidence: f64,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FreeDialogueCandidateEvidence {
    pub bounds: ElementBounds,
    pub direction: AdjacentDirection,
    pub distance_px: f64,
    pub score: f64,
    pub score_threshold: f64,
    pub room: FreeDialogueRoomEvidence,
    pub pixels: FreeDialoguePixelEvidence,
    pub contrast: FreeDialogueContrastEvidence,
    pub association: FreeDialogueAssociationEvidence,
    pub accepted: bool,
    pub rejection_reasons: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SourceBoundInterjectionFallbackEvidence {
    pub exact_gate: &'static str,
    pub layout_strategy: &'static str,
    pub source_region_id: EntityId,
    pub source_bounds: ElementBounds,
    pub source_text: String,
    pub target_translation: String,
    pub target_grapheme_units: usize,
    pub minimum_font_px: f64,
    pub minimum_glyph_height_px: f64,
    pub all_edge_clearance_px: f64,
    pub required_outer_width_px: f64,
    pub required_outer_height_px: f64,
    pub usable_width_px: f64,
    pub usable_height_px: f64,
    pub geometry_can_fit: bool,
    pub native_rerender_required: bool,
    pub reason: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FreeDialogueAnchorAssessment {
    pub detector_producer: &'static str,
    pub detector_version: &'static str,
    pub detector_input: &'static str,
    pub source_region_id: EntityId,
    pub source_bounds: ElementBounds,
    pub role_gate: FreeDialogueRoleGate,
    pub writing_mode_gate: FreeDialogueWritingModeGate,
    pub selected_candidate: Option<FreeDialogueCandidateEvidence>,
    pub source_bound_fallback: Option<SourceBoundInterjectionFallbackEvidence>,
    pub candidates: Vec<FreeDialogueCandidateEvidence>,
    pub rejection_reasons: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FailedFreeDialogueAnchorEvidence {
    pub source_region_id: EntityId,
    pub source_bounds: ElementBounds,
    pub role_gate: FreeDialogueRoleGate,
    pub writing_mode_gate: FreeDialogueWritingModeGate,
    pub candidate_count: usize,
    pub rejection_reason_counts: BTreeMap<&'static str, usize>,
    pub best_rejected_candidates: Vec<FreeDialogueCandidateEvidence>,
    pub source_bound_fallback: Option<SourceBoundInterjectionFallbackEvidence>,
    pub terminal_reason: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FreeDialogueAnchorDecision {
    pub schema_version: u32,
    pub decision: &'static str,
    pub evidence_revision: Revision,
    pub decision_revision: Revision,
    pub page_id: EntityId,
    pub element_id: EntityId,
    pub content_id: EntityId,
    pub source_region_id: EntityId,
    pub target_region_id: EntityId,
    pub source_ocr: SemanticText,
    pub target_translation: SemanticText,
    pub source_bounds: ElementBounds,
    pub candidate_bounds: ElementBounds,
    pub distance_px: f64,
    pub direction: AdjacentDirection,
    pub pixel_analysis: FreeDialoguePixelEvidence,
    pub contrast_background_evidence: FreeDialogueContrastEvidence,
    pub room_evidence: FreeDialogueRoomEvidence,
    pub association: FreeDialogueAssociationEvidence,
    pub writing_modes: FreeDialogueWritingModeGate,
    pub role_gate: FreeDialogueRoleGate,
    pub deterministic_score: f64,
    pub deterministic_score_threshold: f64,
    pub association_confidence: f64,
    pub association_reason: String,
    pub detector_producer: &'static str,
    pub detector_version: &'static str,
    pub detector_input: &'static str,
    pub rejected_candidates: Vec<FreeDialogueCandidateEvidence>,
}

pub(crate) struct FreeDialogueAnchorInput<'a> {
    pub source_region_id: EntityId,
    pub source_bounds: ElementBounds,
    pub source_text: &'a str,
    pub target_text: &'a str,
    pub source_scene_role: &'a str,
    pub required: bool,
    pub has_finite_verified_container_relation: bool,
    pub source_language: Option<&'a str>,
    pub target_language: Option<&'a str>,
    pub source_writing_mode: Option<WritingMode>,
    pub target_writing_mode: Option<WritingMode>,
    pub target_ink_luma: u8,
    pub other_source_bounds: &'a [ElementBounds],
}

pub(crate) fn assess_free_dialogue_anchor(
    image: &GrayImage,
    input: FreeDialogueAnchorInput<'_>,
) -> FreeDialogueAnchorAssessment {
    let scene_role_permits_classification = matches!(
        input.source_scene_role,
        DIALOGUE_ROLE | FREE_DIALOGUE_SOURCE_ROLE
    );
    let compact_source_geometry = input.source_bounds.width <= 40.0
        && input.source_bounds.height <= 72.0
        && input.source_bounds.height >= input.source_bounds.width
        && input.source_bounds.width >= 4.0
        && input.source_bounds.height >= MINIMUM_FONT_PX;
    let role_passed = input.required
        && !input.has_finite_verified_container_relation
        && scene_role_permits_classification
        && compact_source_geometry
        && visible_units(input.source_text) > 0;
    let role_gate = FreeDialogueRoleGate {
        source_scene_role: input.source_scene_role.to_owned(),
        classified_role: "required_uncontained_compact_text",
        exact_gate: "required && source_role in {dialogue,free_text} && nonempty_source_evidence && compact_vertical_source_geometry && no_finite_verified_bubble_or_text_safe_relation",
        source_text: input.source_text.to_owned(),
        required: input.required,
        no_finite_verified_container_relation: !input.has_finite_verified_container_relation,
        source_bounds: input.source_bounds,
        compact_vertical_source_geometry: compact_source_geometry,
        eligible_uncontained_source: role_passed,
        reason: if role_passed {
            "required compact uncontained source role gate passed".to_owned()
        } else {
            "source role, required status, compact geometry, or container relation rejected eligibility"
                .to_owned()
        },
    };
    let writing_passed = input.source_language.is_some_and(is_japanese)
        && input.target_language.is_some_and(is_korean)
        && input.source_writing_mode == Some(WritingMode::Vertical)
        && input.target_writing_mode == Some(WritingMode::Horizontal);
    let writing_mode_gate = FreeDialogueWritingModeGate {
        source_language: input.source_language.unwrap_or_default().to_owned(),
        target_language: input.target_language.unwrap_or_default().to_owned(),
        source_writing_mode: input.source_writing_mode,
        target_writing_mode: input.target_writing_mode,
        exact_gate: "Japanese vertical source to Korean horizontal target",
        passed: writing_passed,
    };
    let mut assessment = FreeDialogueAnchorAssessment {
        detector_producer: RASTER_FREE_DIALOGUE_DETECTOR,
        detector_version: RASTER_FREE_DIALOGUE_DETECTOR_VERSION,
        detector_input: "original_source_pixels",
        source_region_id: input.source_region_id,
        source_bounds: input.source_bounds,
        role_gate,
        writing_mode_gate,
        selected_candidate: None,
        source_bound_fallback: None,
        candidates: Vec::new(),
        rejection_reasons: Vec::new(),
    };
    if !valid_raster_bounds(image, input.source_bounds) {
        assessment
            .rejection_reasons
            .push("source_bounds_are_not_finite_inside_original_raster");
        return assessment;
    }
    if !role_passed {
        assessment.rejection_reasons.push("role_gate_rejected");
        return assessment;
    }
    if !writing_passed {
        assessment
            .rejection_reasons
            .push("writing_mode_or_language_gate_rejected");
        return assessment;
    }
    let target_units = visible_units(input.target_text);
    if target_units == 0 || target_units > 6 {
        assessment
            .rejection_reasons
            .push("target_length_exceeds_compact_anchor_capacity");
        return assessment;
    }

    let required_width = REQUIRED_CLEARANCE_PX * 2.0 + MINIMUM_FONT_PX * target_units as f64;
    let required_height = REQUIRED_CLEARANCE_PX * 2.0 + MINIMUM_FONT_PX;
    let maximum_distance = maximum_source_edge_distance(input.source_bounds);
    let mut candidates = candidate_bounds(
        input.source_bounds,
        required_width,
        required_height,
        maximum_distance,
    );
    candidates.retain(|(_, bounds)| valid_raster_bounds(image, *bounds));
    for (direction, bounds) in candidates {
        assessment.candidates.push(measure_candidate(
            image,
            input.source_bounds,
            bounds,
            direction,
            maximum_distance,
            target_units,
            input.target_ink_luma,
            input.other_source_bounds,
        ));
    }
    assessment.candidates.sort_by(|first, second| {
        second
            .accepted
            .cmp(&first.accepted)
            .then_with(|| second.score.total_cmp(&first.score))
            .then_with(|| first.distance_px.total_cmp(&second.distance_px))
            .then_with(|| direction_rank(first.direction).cmp(&direction_rank(second.direction)))
            .then_with(|| first.bounds.y.total_cmp(&second.bounds.y))
            .then_with(|| first.bounds.x.total_cmp(&second.bounds.x))
    });
    assessment.selected_candidate = assessment
        .candidates
        .iter()
        .find(|candidate| candidate.accepted)
        .cloned();
    if assessment.selected_candidate.is_none() {
        assessment.source_bound_fallback = source_bound_interjection_fallback(
            input.source_region_id,
            input.source_bounds,
            input.source_text,
            input.target_text,
            target_units,
        );
        assessment
            .rejection_reasons
            .push(NO_ADJACENT_CANDIDATE_PASSED_EVIDENCE);
    }
    assessment
}

pub(crate) fn validate_source_bound_interjection_fallback<'a>(
    assessment: &'a FreeDialogueAnchorAssessment,
    source_region_id: EntityId,
    source_bounds: ElementBounds,
    source_text: &str,
    target_text: &str,
) -> Result<&'a SourceBoundInterjectionFallbackEvidence, &'static str> {
    let evidence = assessment
        .source_bound_fallback
        .as_ref()
        .ok_or("source_bound_interjection_fallback_was_not_evidence_eligible")?;
    if assessment.selected_candidate.is_some()
        || !assessment
            .rejection_reasons
            .contains(&NO_ADJACENT_CANDIDATE_PASSED_EVIDENCE)
        || assessment
            .candidates
            .iter()
            .any(|candidate| candidate.accepted)
        || assessment.source_region_id != source_region_id
        || assessment.source_bounds != source_bounds
        || assessment.role_gate.source_text != source_text
        || evidence.source_region_id != source_region_id
        || evidence.source_bounds != source_bounds
        || assessment.role_gate.source_bounds != source_bounds
        || assessment.role_gate.classified_role != "required_uncontained_compact_text"
        || !assessment.role_gate.required
        || !assessment.role_gate.no_finite_verified_container_relation
        || !assessment.role_gate.compact_vertical_source_geometry
        || !assessment.role_gate.eligible_uncontained_source
        || !assessment.writing_mode_gate.passed
        || source_text != evidence.source_text
        || target_text != evidence.target_translation
    {
        return Err("source_bound_interjection_fallback_semantic_identity_mismatch");
    }
    let target_units = visible_units(target_text);
    let required_width = REQUIRED_CLEARANCE_PX * 2.0 + MINIMUM_FONT_PX;
    let required_height = REQUIRED_CLEARANCE_PX * 2.0 + MINIMUM_FONT_PX * target_units as f64;
    if !evidence.geometry_can_fit
        || target_units == 0
        || target_units > MAXIMUM_SOURCE_BOUND_FALLBACK_TARGET_UNITS
        || required_width > source_bounds.width + f64::EPSILON
        || required_height > source_bounds.height + f64::EPSILON
        || evidence.layout_strategy != "native_vertical_korean_in_source_region"
        || !evidence.native_rerender_required
    {
        return Err("source_bound_interjection_fallback_room_or_strategy_mismatch");
    }
    Ok(evidence)
}

pub(crate) fn has_recorded_no_adjacent_candidate_outcome(
    assessment: &FreeDialogueAnchorAssessment,
) -> bool {
    assessment.selected_candidate.is_none()
        && assessment
            .rejection_reasons
            .contains(&NO_ADJACENT_CANDIDATE_PASSED_EVIDENCE)
        && assessment
            .candidates
            .iter()
            .all(|candidate| !candidate.accepted)
}

pub(crate) fn validate_free_dialogue_anchor_decision(
    decision: &FreeDialogueAnchorDecision,
    element_id: EntityId,
    content_id: EntityId,
    source_region_id: EntityId,
    source_bounds: ElementBounds,
    source_text: &str,
    target_text: &str,
) -> Result<(), &'static str> {
    if decision.schema_version != FREE_DIALOGUE_ANCHOR_SCHEMA_VERSION
        || decision.element_id != element_id
        || decision.content_id != content_id
        || decision.source_region_id != source_region_id
        || decision.source_bounds != source_bounds
        || decision.role_gate.source_bounds != source_bounds
        || decision.role_gate.source_text != source_text
        || decision.source_ocr.text != source_text
    {
        return Err("free_dialogue_anchor_semantic_identity_mismatch");
    }
    if visible_units(target_text) == 0
        || visible_units(target_text) > decision.room_evidence.target_grapheme_units
    {
        return Err("free_dialogue_anchor_target_length_exceeds_pixel_proven_room");
    }
    if decision.detector_producer != RASTER_FREE_DIALOGUE_DETECTOR
        || decision.detector_version != RASTER_FREE_DIALOGUE_DETECTOR_VERSION
        || decision.detector_input != "original_source_pixels"
    {
        return Err("free_dialogue_anchor_detector_provenance_mismatch");
    }
    if decision.role_gate.classified_role != "required_uncontained_compact_text"
        || !decision.role_gate.required
        || !decision.role_gate.no_finite_verified_container_relation
        || !decision.role_gate.compact_vertical_source_geometry
        || !decision.role_gate.eligible_uncontained_source
        || !decision.writing_modes.passed
    {
        return Err("free_dialogue_anchor_role_or_writing_gate_mismatch");
    }
    let measured_edge_distance = rectangle_edge_distance(source_bounds, decision.candidate_bounds);
    let maximum_edge_distance = maximum_source_edge_distance(source_bounds);
    let association = &decision.association;
    let attribution_measurements_match = match (
        association.nearest_other_source_center_distance_px,
        association.attribution_margin_px,
    ) {
        (Some(nearest_other), Some(margin)) => approximately_equal(
            margin,
            nearest_other - association.source_center_distance_px,
        ),
        (None, None) => true,
        _ => false,
    };
    if !approximately_equal(decision.distance_px, measured_edge_distance)
        || !approximately_equal(association.source_edge_distance_px, measured_edge_distance)
        || !approximately_equal(
            association.maximum_source_edge_distance_px,
            maximum_edge_distance,
        )
        || !approximately_equal(
            association.source_center_distance_px,
            distance(center(source_bounds), center(decision.candidate_bounds)),
        )
        || association.source_edge_distance_within_cap
            != (measured_edge_distance <= maximum_edge_distance)
        || association.has_required_attribution_margin
            != association
                .attribution_margin_px
                .is_none_or(|margin| margin >= REQUIRED_CLEARANCE_PX)
        || !attribution_measurements_match
        || association.preserves_reading_order_and_attribution
            != (association.source_edge_distance_within_cap
                && association.has_required_attribution_margin
                && association.reading_order_preserved)
    {
        return Err("free_dialogue_anchor_association_contract_mismatch");
    }
    if decision.deterministic_score < decision.deterministic_score_threshold
        || decision.deterministic_score_threshold != MINIMUM_FREE_DIALOGUE_ANCHOR_SCORE
        || !decision.room_evidence.fits_target_length
        || !decision.contrast_background_evidence.contrastable
        || !decision.association.preserves_reading_order_and_attribution
        || decision.pixel_analysis.overlaps_existing_text_bounds
        || decision.pixel_analysis.edge_density > MAX_EDGE_DENSITY
        || decision.pixel_analysis.dark_ink_ratio > MAX_DARK_INK_RATIO
        || decision.pixel_analysis.luma_standard_deviation > MAX_LUMA_STANDARD_DEVIATION
        || decision.pixel_analysis.boundary_edge_density > MAX_BOUNDARY_EDGE_DENSITY
        || decision.pixel_analysis.boundary_max_axis_edge_coverage > MAX_BOUNDARY_AXIS_COVERAGE
    {
        return Err("free_dialogue_anchor_raster_evidence_below_threshold");
    }
    Ok(())
}

pub(crate) fn failed_free_dialogue_anchor_evidence(
    assessment: &FreeDialogueAnchorAssessment,
) -> Option<FailedFreeDialogueAnchorEvidence> {
    if assessment.selected_candidate.is_some()
        || !assessment.role_gate.required
        || !assessment.role_gate.eligible_uncontained_source
        || assessment.role_gate.classified_role != "required_uncontained_compact_text"
    {
        return None;
    }
    let mut rejection_reason_counts = BTreeMap::new();
    for candidate in &assessment.candidates {
        for reason in &candidate.rejection_reasons {
            *rejection_reason_counts.entry(*reason).or_insert(0) += 1;
        }
    }
    Some(FailedFreeDialogueAnchorEvidence {
        source_region_id: assessment.source_region_id,
        source_bounds: assessment.source_bounds,
        role_gate: assessment.role_gate.clone(),
        writing_mode_gate: assessment.writing_mode_gate.clone(),
        candidate_count: assessment.candidates.len(),
        rejection_reason_counts,
        best_rejected_candidates: assessment.candidates.iter().take(3).cloned().collect(),
        source_bound_fallback: assessment.source_bound_fallback.clone(),
        terminal_reason: if assessment.source_bound_fallback.is_some() {
            "no adjacent candidate passed every unchanged source-pixel, text-overlap, boundary, room, contrast, score, reading-order, and attribution gate; the source-bound fallback is only preview-eligible until a native deterministic rerender passes"
        } else {
            "no adjacent candidate passed every unchanged source-pixel, text-overlap, boundary, room, contrast, score, reading-order, and attribution gate, and the original source bounds cannot geometrically contain the narrow vertical fallback"
        },
    })
}

fn measure_candidate(
    image: &GrayImage,
    source: ElementBounds,
    bounds: ElementBounds,
    direction: AdjacentDirection,
    maximum_distance: f64,
    target_units: usize,
    target_ink_luma: u8,
    other_source_bounds: &[ElementBounds],
) -> FreeDialogueCandidateEvidence {
    let distance_px = rectangle_edge_distance(source, bounds);
    let room = FreeDialogueRoomEvidence {
        target_grapheme_units: target_units,
        minimum_font_px: MINIMUM_FONT_PX,
        minimum_glyph_height_px: MINIMUM_GLYPH_HEIGHT_PX,
        all_edge_clearance_px: REQUIRED_CLEARANCE_PX,
        required_outer_width_px: REQUIRED_CLEARANCE_PX * 2.0
            + MINIMUM_FONT_PX * target_units as f64,
        required_outer_height_px: REQUIRED_CLEARANCE_PX * 2.0 + MINIMUM_FONT_PX,
        usable_width_px: bounds.width - REQUIRED_CLEARANCE_PX * 2.0,
        usable_height_px: bounds.height - REQUIRED_CLEARANCE_PX * 2.0,
        fits_target_length: bounds.width + f64::EPSILON
            >= REQUIRED_CLEARANCE_PX * 2.0 + MINIMUM_FONT_PX * target_units as f64
            && bounds.height + f64::EPSILON >= REQUIRED_CLEARANCE_PX * 2.0 + MINIMUM_FONT_PX,
    };
    let overlaps_existing_text_bounds = other_source_bounds
        .iter()
        .any(|existing| intersection_area(expand(*existing, 2.0), bounds) > 0.0);
    let (sampled_pixels, mean_luma, standard_deviation, minimum_luma, maximum_luma, dark_ratio) =
        luma_measurements(image, bounds);
    let edge_density = edge_density(image, bounds);
    let corridor = corridor_bounds(source, bounds, direction);
    let (boundary_sampled_pixels, boundary_edge_density, boundary_max_axis_edge_coverage) =
        boundary_measurements(image, corridor, direction);
    let pixels = FreeDialoguePixelEvidence {
        sampled_pixels,
        mean_luma,
        luma_standard_deviation: standard_deviation,
        minimum_luma,
        maximum_luma,
        dark_ink_ratio: dark_ratio,
        edge_density,
        boundary_sampled_pixels,
        boundary_edge_density,
        boundary_max_axis_edge_coverage,
        overlaps_existing_text_bounds,
    };
    let absolute_contrast = (mean_luma - f64::from(target_ink_luma)).abs();
    let contrast = FreeDialogueContrastEvidence {
        target_ink_luma,
        background_mean_luma: mean_luma,
        absolute_contrast,
        minimum_contrast: MIN_TARGET_INK_CONTRAST,
        contrastable: absolute_contrast >= MIN_TARGET_INK_CONTRAST,
    };
    let source_center = center(source);
    let candidate_center = center(bounds);
    let source_center_distance = distance(source_center, candidate_center);
    let nearest_other = other_source_bounds
        .iter()
        .map(|other| distance(center(*other), candidate_center))
        .min_by(f64::total_cmp);
    let attribution_margin = nearest_other.map(|other| other - source_center_distance);
    let reading_order_preserved = other_source_bounds
        .iter()
        .all(|other| !movement_conflicts_with_other_source_text(source, bounds, *other));
    let source_edge_distance_within_cap = distance_px <= maximum_distance;
    let has_required_attribution_margin =
        attribution_margin.is_none_or(|margin| margin >= REQUIRED_CLEARANCE_PX);
    let preserves_attribution = source_edge_distance_within_cap
        && has_required_attribution_margin
        && reading_order_preserved;
    let distance_score = (1.0 - distance_px / maximum_distance).clamp(0.0, 1.0);
    let attribution_score =
        attribution_margin.map_or(1.0, |margin| (margin / maximum_distance).clamp(0.0, 1.0));
    let reading_order_score = f64::from(reading_order_preserved);
    let association_confidence =
        (0.25 * distance_score + 0.35 * attribution_score + 0.40 * reading_order_score)
            .clamp(0.0, 1.0);
    let association = FreeDialogueAssociationEvidence {
        source_center_distance_px: source_center_distance,
        source_edge_distance_px: distance_px,
        maximum_source_edge_distance_px: maximum_distance,
        source_edge_distance_within_cap,
        nearest_other_source_center_distance_px: nearest_other,
        attribution_margin_px: attribution_margin,
        has_required_attribution_margin,
        reading_order_preserved,
        preserves_reading_order_and_attribution: preserves_attribution,
        confidence: association_confidence,
        reason: if preserves_attribution {
            "adjacent patch edge stays within the source-scaled edge-distance cap and its center remains uniquely attributable to the original source text without changing reading order"
                .to_owned()
        } else {
            "patch edge would exceed the source-scaled edge-distance cap or its independent reading-order/nearest-text attribution gates are ambiguous"
                .to_owned()
        },
    };
    let score = candidate_score(
        standard_deviation,
        edge_density,
        dark_ratio,
        boundary_edge_density,
        boundary_max_axis_edge_coverage,
        absolute_contrast,
        association_confidence,
    );
    let mut rejection_reasons = Vec::new();
    if !room.fits_target_length {
        rejection_reasons.push("insufficient_12px_font_glyph_and_4px_clearance_room");
    }
    if overlaps_existing_text_bounds {
        rejection_reasons.push("overlaps_existing_source_or_translated_text_bounds");
    }
    if edge_density > MAX_EDGE_DENSITY || dark_ratio > MAX_DARK_INK_RATIO {
        rejection_reasons.push("high_edge_density_or_character_face_outline_structure");
    }
    if standard_deviation > MAX_LUMA_STANDARD_DEVIATION {
        rejection_reasons.push("background_is_not_sufficiently_uniform");
    }
    if !contrast.contrastable {
        rejection_reasons.push("background_cannot_contrast_with_target_ink");
    }
    if boundary_edge_density > MAX_BOUNDARY_EDGE_DENSITY
        || boundary_max_axis_edge_coverage > MAX_BOUNDARY_AXIS_COVERAGE
    {
        rejection_reasons.push("candidate_crosses_probable_bubble_or_panel_boundary");
    }
    if !association.preserves_reading_order_and_attribution {
        rejection_reasons.push("reading_order_or_visual_attribution_is_ambiguous");
    }
    if score < MINIMUM_FREE_DIALOGUE_ANCHOR_SCORE {
        rejection_reasons.push("deterministic_text_safe_score_below_threshold");
    }
    FreeDialogueCandidateEvidence {
        bounds,
        direction,
        distance_px,
        score,
        score_threshold: MINIMUM_FREE_DIALOGUE_ANCHOR_SCORE,
        room,
        pixels,
        contrast,
        association,
        accepted: rejection_reasons.is_empty(),
        rejection_reasons,
    }
}

fn candidate_score(
    luma_standard_deviation: f64,
    edge_density: f64,
    dark_ink_ratio: f64,
    boundary_edge_density: f64,
    boundary_max_axis_edge_coverage: f64,
    absolute_contrast: f64,
    association_confidence: f64,
) -> f64 {
    let uniformity_score =
        (1.0 - luma_standard_deviation / MAX_LUMA_STANDARD_DEVIATION).clamp(0.0, 1.0);
    let edge_score = (1.0 - edge_density / MAX_EDGE_DENSITY).clamp(0.0, 1.0);
    let ink_score = (1.0 - dark_ink_ratio / MAX_DARK_INK_RATIO).clamp(0.0, 1.0);
    let boundary_score = (1.0 - boundary_edge_density / MAX_BOUNDARY_EDGE_DENSITY)
        .clamp(0.0, 1.0)
        .min((1.0 - boundary_max_axis_edge_coverage / MAX_BOUNDARY_AXIS_COVERAGE).clamp(0.0, 1.0));
    let contrast_score = (absolute_contrast / 192.0).clamp(0.0, 1.0);
    0.23 * uniformity_score
        + 0.20 * edge_score
        + 0.12 * ink_score
        + 0.15 * boundary_score
        + 0.12 * contrast_score
        + 0.18 * association_confidence
}

fn candidate_bounds(
    source: ElementBounds,
    width: f64,
    height: f64,
    maximum_distance: f64,
) -> Vec<(AdjacentDirection, ElementBounds)> {
    let mut candidates = Vec::new();
    let mut gap = REQUIRED_CLEARANCE_PX;
    while gap <= maximum_distance {
        for direction in [
            AdjacentDirection::Right,
            AdjacentDirection::Left,
            AdjacentDirection::Above,
            AdjacentDirection::Below,
        ] {
            let offsets = match direction {
                AdjacentDirection::Right | AdjacentDirection::Left => {
                    [-source.height * 0.25, 0.0, source.height * 0.25]
                }
                AdjacentDirection::Above | AdjacentDirection::Below => {
                    [-source.width * 0.25, 0.0, source.width * 0.25]
                }
            };
            for offset in offsets {
                let bounds = match direction {
                    AdjacentDirection::Right => ElementBounds {
                        x: source.x + source.width + gap,
                        y: source.y + (source.height - height) * 0.5 + offset,
                        width,
                        height,
                    },
                    AdjacentDirection::Left => ElementBounds {
                        x: source.x - gap - width,
                        y: source.y + (source.height - height) * 0.5 + offset,
                        width,
                        height,
                    },
                    AdjacentDirection::Above => ElementBounds {
                        x: source.x + (source.width - width) * 0.5 + offset,
                        y: source.y - gap - height,
                        width,
                        height,
                    },
                    AdjacentDirection::Below => ElementBounds {
                        x: source.x + (source.width - width) * 0.5 + offset,
                        y: source.y + source.height + gap,
                        width,
                        height,
                    },
                };
                candidates.push((direction, bounds));
            }
        }
        gap += 4.0;
    }
    candidates
}

fn source_bound_interjection_fallback(
    source_region_id: EntityId,
    source_bounds: ElementBounds,
    source_text: &str,
    target_text: &str,
    target_units: usize,
) -> Option<SourceBoundInterjectionFallbackEvidence> {
    if target_units == 0 || target_units > MAXIMUM_SOURCE_BOUND_FALLBACK_TARGET_UNITS {
        return None;
    }
    let required_width = REQUIRED_CLEARANCE_PX * 2.0 + MINIMUM_FONT_PX;
    let required_height = REQUIRED_CLEARANCE_PX * 2.0 + MINIMUM_FONT_PX * target_units as f64;
    let geometry_can_fit = source_bounds.width + f64::EPSILON >= required_width
        && source_bounds.height + f64::EPSILON >= required_height;
    geometry_can_fit.then(|| SourceBoundInterjectionFallbackEvidence {
        exact_gate: "no adjacent candidate passed && required compact no-container source role && Japanese-vertical/Korean target writing gate && target has at most two visible graphemes && original source bounds can contain native vertical rendering at 12px with 4px clearance",
        layout_strategy: "native_vertical_korean_in_source_region",
        source_region_id,
        source_bounds,
        source_text: source_text.to_owned(),
        target_translation: target_text.to_owned(),
        target_grapheme_units: target_units,
        minimum_font_px: MINIMUM_FONT_PX,
        minimum_glyph_height_px: MINIMUM_GLYPH_HEIGHT_PX,
        all_edge_clearance_px: REQUIRED_CLEARANCE_PX,
        required_outer_width_px: required_width,
        required_outer_height_px: required_height,
        usable_width_px: source_bounds.width - REQUIRED_CLEARANCE_PX * 2.0,
        usable_height_px: source_bounds.height - REQUIRED_CLEARANCE_PX * 2.0,
        geometry_can_fit,
        native_rerender_required: true,
        reason: "geometry is eligible only for a controlled native-renderer preview; commit still requires every deterministic rerender gate",
    })
}

fn maximum_source_edge_distance(source: ElementBounds) -> f64 {
    source
        .width
        .max(source.height)
        .mul_add(1.5, REQUIRED_CLEARANCE_PX * 2.0)
        .clamp(24.0, 96.0)
}

fn rectangle_edge_distance(first: ElementBounds, second: ElementBounds) -> f64 {
    let horizontal_gap = (first.x - second.x - second.width)
        .max(second.x - first.x - first.width)
        .max(0.0);
    let vertical_gap = (first.y - second.y - second.height)
        .max(second.y - first.y - first.height)
        .max(0.0);
    horizontal_gap.hypot(vertical_gap)
}

fn approximately_equal(first: f64, second: f64) -> bool {
    (first - second).abs() <= 1e-9
}

fn visible_units(text: &str) -> usize {
    text.graphemes(true)
        .filter(|unit| {
            unit.chars()
                .any(|character| !character.is_whitespace() && !character.is_control())
        })
        .count()
}

fn luma_measurements(image: &GrayImage, bounds: ElementBounds) -> (u32, f64, f64, u8, u8, f64) {
    let (min_x, min_y, max_x, max_y) = integer_bounds(image, bounds);
    let mut count = 0_u32;
    let mut sum = 0.0;
    let mut squared_sum = 0.0;
    let mut minimum = u8::MAX;
    let mut maximum = u8::MIN;
    let mut dark = 0_u32;
    for y in min_y..max_y {
        for x in min_x..max_x {
            let value = image.get_pixel(x, y)[0];
            let numeric = f64::from(value);
            count += 1;
            sum += numeric;
            squared_sum += numeric * numeric;
            minimum = minimum.min(value);
            maximum = maximum.max(value);
            dark += u32::from(value <= DARK_LUMA);
        }
    }
    if count == 0 {
        return (0, 0.0, f64::INFINITY, 0, 0, 1.0);
    }
    let mean = sum / f64::from(count);
    let variance = (squared_sum / f64::from(count) - mean * mean).max(0.0);
    (
        count,
        mean,
        variance.sqrt(),
        minimum,
        maximum,
        f64::from(dark) / f64::from(count),
    )
}

fn edge_density(image: &GrayImage, bounds: ElementBounds) -> f64 {
    let (min_x, min_y, max_x, max_y) = integer_bounds(image, bounds);
    let mut edges = 0_u32;
    let mut samples = 0_u32;
    for y in min_y..max_y.saturating_sub(1) {
        for x in min_x..max_x.saturating_sub(1) {
            let value = i16::from(image.get_pixel(x, y)[0]);
            let horizontal = (value - i16::from(image.get_pixel(x + 1, y)[0])).abs();
            let vertical = (value - i16::from(image.get_pixel(x, y + 1)[0])).abs();
            edges += u32::from(horizontal > EDGE_DELTA || vertical > EDGE_DELTA);
            samples += 1;
        }
    }
    f64::from(edges) / f64::from(samples.max(1))
}

fn boundary_measurements(
    image: &GrayImage,
    bounds: ElementBounds,
    direction: AdjacentDirection,
) -> (u32, f64, f64) {
    if !valid_raster_bounds(image, bounds) || bounds.width < 1.0 || bounds.height < 1.0 {
        return (0, 1.0, 1.0);
    }
    let density = edge_density(image, bounds);
    let (min_x, min_y, max_x, max_y) = integer_bounds(image, bounds);
    let mut maximum: f64 = 0.0;
    match direction {
        AdjacentDirection::Right | AdjacentDirection::Left => {
            for x in min_x..max_x.saturating_sub(1) {
                let mut edges = 0_u32;
                let mut samples = 0_u32;
                for y in min_y..max_y.saturating_sub(1) {
                    let current = i16::from(image.get_pixel(x, y)[0]);
                    let next = i16::from(image.get_pixel(x + 1, y)[0]);
                    edges += u32::from((current - next).abs() > EDGE_DELTA);
                    samples += 1;
                }
                maximum = maximum.max(f64::from(edges) / f64::from(samples.max(1)));
            }
        }
        AdjacentDirection::Above | AdjacentDirection::Below => {
            for y in min_y..max_y.saturating_sub(1) {
                let mut edges = 0_u32;
                let mut samples = 0_u32;
                for x in min_x..max_x.saturating_sub(1) {
                    let current = i16::from(image.get_pixel(x, y)[0]);
                    let next = i16::from(image.get_pixel(x, y + 1)[0]);
                    edges += u32::from((current - next).abs() > EDGE_DELTA);
                    samples += 1;
                }
                maximum = maximum.max(f64::from(edges) / f64::from(samples.max(1)));
            }
        }
    }
    let sampled = ((max_x - min_x) * (max_y - min_y)) as u32;
    (sampled, density, maximum)
}

fn corridor_bounds(
    source: ElementBounds,
    candidate: ElementBounds,
    direction: AdjacentDirection,
) -> ElementBounds {
    match direction {
        AdjacentDirection::Right => ElementBounds {
            x: source.x + source.width,
            y: source.y.max(candidate.y),
            width: candidate.x - source.x - source.width,
            height: (source.y + source.height).min(candidate.y + candidate.height)
                - source.y.max(candidate.y),
        },
        AdjacentDirection::Left => ElementBounds {
            x: candidate.x + candidate.width,
            y: source.y.max(candidate.y),
            width: source.x - candidate.x - candidate.width,
            height: (source.y + source.height).min(candidate.y + candidate.height)
                - source.y.max(candidate.y),
        },
        AdjacentDirection::Above => ElementBounds {
            x: source.x.max(candidate.x),
            y: candidate.y + candidate.height,
            width: (source.x + source.width).min(candidate.x + candidate.width)
                - source.x.max(candidate.x),
            height: source.y - candidate.y - candidate.height,
        },
        AdjacentDirection::Below => ElementBounds {
            x: source.x.max(candidate.x),
            y: source.y + source.height,
            width: (source.x + source.width).min(candidate.x + candidate.width)
                - source.x.max(candidate.x),
            height: candidate.y - source.y - source.height,
        },
    }
}

fn integer_bounds(image: &GrayImage, bounds: ElementBounds) -> (u32, u32, u32, u32) {
    (
        bounds.x.floor().max(0.0) as u32,
        bounds.y.floor().max(0.0) as u32,
        (bounds.x + bounds.width)
            .ceil()
            .min(f64::from(image.width())) as u32,
        (bounds.y + bounds.height)
            .ceil()
            .min(f64::from(image.height())) as u32,
    )
}

fn valid_raster_bounds(image: &GrayImage, bounds: ElementBounds) -> bool {
    [bounds.x, bounds.y, bounds.width, bounds.height]
        .into_iter()
        .all(f64::is_finite)
        && bounds.x >= 0.0
        && bounds.y >= 0.0
        && bounds.width > 0.0
        && bounds.height > 0.0
        && bounds.x + bounds.width <= f64::from(image.width())
        && bounds.y + bounds.height <= f64::from(image.height())
}

fn expand(bounds: ElementBounds, amount: f64) -> ElementBounds {
    ElementBounds {
        x: bounds.x - amount,
        y: bounds.y - amount,
        width: bounds.width + amount * 2.0,
        height: bounds.height + amount * 2.0,
    }
}

fn intersection_area(first: ElementBounds, second: ElementBounds) -> f64 {
    let width =
        ((first.x + first.width).min(second.x + second.width) - first.x.max(second.x)).max(0.0);
    let height =
        ((first.y + first.height).min(second.y + second.height) - first.y.max(second.y)).max(0.0);
    width * height
}

fn center(bounds: ElementBounds) -> (f64, f64) {
    (
        bounds.x + bounds.width * 0.5,
        bounds.y + bounds.height * 0.5,
    )
}

fn distance(first: (f64, f64), second: (f64, f64)) -> f64 {
    (first.0 - second.0).hypot(first.1 - second.1)
}

fn movement_conflicts_with_other_source_text(
    source: ElementBounds,
    candidate: ElementBounds,
    other: ElementBounds,
) -> bool {
    let relocation_slot = ElementBounds {
        x: source.x.min(candidate.x),
        y: source.y.min(candidate.y),
        width: (source.x + source.width).max(candidate.x + candidate.width)
            - source.x.min(candidate.x),
        height: (source.y + source.height).max(candidate.y + candidate.height)
            - source.y.min(candidate.y),
    };
    intersection_area(relocation_slot, expand(other, 2.0)) > 0.0
}

fn direction_rank(direction: AdjacentDirection) -> u8 {
    match direction {
        AdjacentDirection::Right => 0,
        AdjacentDirection::Left => 1,
        AdjacentDirection::Above => 2,
        AdjacentDirection::Below => 3,
    }
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

#[cfg(test)]
mod tests {
    use image::{GrayImage, Luma};

    use super::*;

    fn source() -> ElementBounds {
        ElementBounds {
            x: 60.0,
            y: 50.0,
            width: 16.0,
            height: 40.0,
        }
    }

    fn input<'a>(other_source_bounds: &'a [ElementBounds]) -> FreeDialogueAnchorInput<'a> {
        FreeDialogueAnchorInput {
            source_region_id: EntityId::new(),
            source_bounds: source(),
            source_text: "え〜",
            target_text: "에~",
            source_scene_role: FREE_DIALOGUE_SOURCE_ROLE,
            required: true,
            has_finite_verified_container_relation: false,
            source_language: Some("ja-JP"),
            target_language: Some("ko-KR"),
            source_writing_mode: Some(WritingMode::Vertical),
            target_writing_mode: Some(WritingMode::Horizontal),
            target_ink_luma: 0,
            other_source_bounds,
        }
    }

    #[test]
    fn source_edge_cap_is_independent_from_center_attribution_for_reported_geometry() {
        let image = GrayImage::from_pixel(1_000, 1_200, Luma([255]));
        let source = ElementBounds {
            x: 472.166_170_167_871_9,
            y: 805.619_672_014_744,
            width: 17.425_472_164_256_234,
            height: 38.760_655_970_512_06,
        };
        let candidate = ElementBounds {
            x: 460.522_538_208_935_94,
            y: 721.619_672_014_744,
            width: 32.0,
            height: 20.0,
        };
        let maximum_distance = maximum_source_edge_distance(source);
        let candidate_center = center(candidate);
        let safely_distant_other_center_y = candidate_center.1 + 181.878_550_489_200_85;
        let safely_distant_other = ElementBounds {
            x: candidate_center.0 - 4.0,
            y: safely_distant_other_center_y - 4.0,
            width: 8.0,
            height: 8.0,
        };

        let accepted = measure_candidate(
            &image,
            source,
            candidate,
            AdjacentDirection::Above,
            maximum_distance,
            2,
            0,
            &[safely_distant_other],
        );
        assert!(approximately_equal(accepted.distance_px, 64.0));
        assert!(approximately_equal(
            accepted.association.maximum_source_edge_distance_px,
            66.140_983_955_768_09
        ));
        assert!(
            accepted.association.source_center_distance_px
                > accepted.association.maximum_source_edge_distance_px
        );
        assert!(accepted.association.source_edge_distance_within_cap);
        assert!(accepted.association.has_required_attribution_margin);
        assert!(accepted.association.reading_order_preserved);
        assert!(accepted.room.fits_target_length);
        assert!(!accepted.pixels.overlaps_existing_text_bounds);
        assert_eq!(accepted.pixels.dark_ink_ratio, 0.0);
        assert_eq!(accepted.pixels.edge_density, 0.0);
        assert_eq!(accepted.pixels.boundary_edge_density, 0.0);
        assert!(accepted.contrast.contrastable);
        assert!(accepted.score >= MINIMUM_FREE_DIALOGUE_ANCHOR_SCORE);
        assert!(accepted.accepted, "{:?}", accepted.rejection_reasons);

        let outside_cap = ElementBounds {
            y: source.y - candidate.height - maximum_distance - 1.0,
            ..candidate
        };
        let outside_cap = measure_candidate(
            &image,
            source,
            outside_cap,
            AdjacentDirection::Above,
            maximum_distance,
            2,
            0,
            &[],
        );
        assert!(!outside_cap.association.source_edge_distance_within_cap);
        assert!(!outside_cap.accepted);
        assert!(
            outside_cap
                .rejection_reasons
                .contains(&"reading_order_or_visual_attribution_is_ambiguous")
        );

        let insufficient_margin_other_center_x =
            candidate_center.0 + accepted.association.source_center_distance_px + 2.0;
        let insufficient_margin_other = ElementBounds {
            x: insufficient_margin_other_center_x - 4.0,
            y: candidate_center.1 - 4.0,
            width: 8.0,
            height: 8.0,
        };
        let insufficient_margin = measure_candidate(
            &image,
            source,
            candidate,
            AdjacentDirection::Above,
            maximum_distance,
            2,
            0,
            &[insufficient_margin_other],
        );
        assert!(
            insufficient_margin
                .association
                .source_edge_distance_within_cap
        );
        assert!(insufficient_margin.association.reading_order_preserved);
        assert!(
            !insufficient_margin
                .association
                .has_required_attribution_margin
        );
        assert!(insufficient_margin.score >= MINIMUM_FREE_DIALOGUE_ANCHOR_SCORE);
        assert!(!insufficient_margin.accepted);
        assert!(
            insufficient_margin
                .rejection_reasons
                .contains(&"reading_order_or_visual_attribution_is_ambiguous")
        );
    }

    #[test]
    fn exact_edge_anchor_uses_other_text_conflicts_instead_of_global_axis_reordering() {
        let image = GrayImage::from_pixel(1_000, 1_200, Luma([255]));
        let source = ElementBounds {
            x: 472.166_170_167_871_9,
            y: 805.619_672_014_744,
            width: 17.425_472_164_256_234,
            height: 38.760_655_970_512_06,
        };
        let candidate = ElementBounds {
            x: 460.522_538_208_935_94,
            y: 721.619_672_014_744,
            width: 32.0,
            height: 20.0,
        };
        // This text center is vertically between source and candidate, which made the old global
        // side-of-center rule report a reorder. Its pixels and semantic slot are far to the right.
        let distant_same_panel_text = ElementBounds {
            x: 647.848_380_548_540_4,
            y: 776.0,
            width: 8.0,
            height: 8.0,
        };

        let measured = measure_candidate(
            &image,
            source,
            candidate,
            AdjacentDirection::Above,
            maximum_source_edge_distance(source),
            2,
            0,
            &[distant_same_panel_text],
        );

        assert!(approximately_equal(measured.distance_px, 64.0));
        assert!(measured.association.source_edge_distance_within_cap);
        assert!(
            measured
                .association
                .nearest_other_source_center_distance_px
                .unwrap()
                > measured.association.source_center_distance_px + REQUIRED_CLEARANCE_PX
        );
        assert!(approximately_equal(
            measured
                .association
                .nearest_other_source_center_distance_px
                .unwrap(),
            181.878_550_489_200_85
        ));
        assert!(approximately_equal(
            measured.association.attribution_margin_px.unwrap(),
            88.396_661_349_415_08
        ));
        assert!(measured.association.reading_order_preserved);
        assert!(measured.association.preserves_reading_order_and_attribution);
        assert!(!measured.pixels.overlaps_existing_text_bounds);
        assert!(measured.accepted, "{:?}", measured.rejection_reasons);
        assert!(
            candidate_score(
                0.0,
                0.0,
                0.0,
                0.090_992_647_058_823_53,
                0.411_764_705_882_352_9,
                255.0,
                measured.association.confidence,
            ) >= MINIMUM_FREE_DIALOGUE_ANCHOR_SCORE
        );

        let actual_ordering_conflict = ElementBounds {
            x: 490.0,
            y: 770.0,
            width: 200.0,
            height: 8.0,
        };
        let conflict = measure_candidate(
            &image,
            source,
            candidate,
            AdjacentDirection::Above,
            maximum_source_edge_distance(source),
            2,
            0,
            &[actual_ordering_conflict],
        );
        assert!(!conflict.pixels.overlaps_existing_text_bounds);
        assert!(conflict.association.has_required_attribution_margin);
        assert!(!conflict.association.reading_order_preserved);
        assert!(!conflict.accepted);
        assert!(
            conflict
                .rejection_reasons
                .contains(&"reading_order_or_visual_attribution_is_ambiguous")
        );
    }

    #[test]
    fn fitting_source_geometry_exposes_only_the_evidence_bound_vertical_source_fallback() {
        let source_bounds = ElementBounds {
            x: 673.572_784_315_471_2,
            y: 870.776_877_422_549_4,
            width: 25.846_618_869_057_693,
            height: 49.071_245_154_901_135,
        };
        let unsafe_right_bounds = ElementBounds {
            x: 707.419_403_184_528_8,
            y: 873.044_688_711_274_7,
            width: 32.0,
            height: 20.0,
        };
        let mut boundary_image = GrayImage::from_pixel(849, 1_200, Luma([255]));
        for y in 873..894 {
            for x in 699..703 {
                boundary_image.put_pixel(x, y, Luma([0]));
            }
        }
        let unsafe_right = measure_candidate(
            &boundary_image,
            source_bounds,
            unsafe_right_bounds,
            AdjacentDirection::Right,
            maximum_source_edge_distance(source_bounds),
            2,
            0,
            &[],
        );
        assert_eq!(unsafe_right.bounds, unsafe_right_bounds);
        assert_eq!(unsafe_right.pixels.edge_density, 0.0);
        assert!(unsafe_right.pixels.boundary_max_axis_edge_coverage > MAX_BOUNDARY_AXIS_COVERAGE);
        assert!(!unsafe_right.accepted);
        assert!(
            unsafe_right
                .rejection_reasons
                .contains(&"candidate_crosses_probable_bubble_or_panel_boundary")
        );

        let image = GrayImage::from_pixel(849, 1_200, Luma([0]));
        let mut candidate = input(&[]);
        candidate.source_bounds = source_bounds;
        candidate.source_text = "ほう";
        candidate.target_text = "허.";
        let assessment = assess_free_dialogue_anchor(&image, candidate);

        assert!(assessment.selected_candidate.is_none());
        assert!(has_recorded_no_adjacent_candidate_outcome(&assessment));
        let fallback = assessment.source_bound_fallback.as_ref().unwrap();
        assert_eq!(
            fallback.layout_strategy,
            "native_vertical_korean_in_source_region"
        );
        assert_eq!(fallback.source_bounds, source_bounds);
        assert_eq!(fallback.target_grapheme_units, 2);
        assert_eq!(fallback.required_outer_width_px, 20.0);
        assert_eq!(fallback.required_outer_height_px, 32.0);
        assert!(fallback.geometry_can_fit);
        assert!(fallback.native_rerender_required);
        assert!(
            validate_source_bound_interjection_fallback(
                &assessment,
                assessment.source_region_id,
                source_bounds,
                "ほう",
                "허.",
            )
            .is_ok()
        );
        let mut missing_outcome = assessment.clone();
        missing_outcome
            .rejection_reasons
            .retain(|reason| *reason != NO_ADJACENT_CANDIDATE_PASSED_EVIDENCE);
        assert!(
            validate_source_bound_interjection_fallback(
                &missing_outcome,
                missing_outcome.source_region_id,
                source_bounds,
                "ほう",
                "허.",
            )
            .is_err()
        );

        let mut longer = input(&[]);
        longer.source_bounds = source_bounds;
        longer.source_text = "ほう";
        longer.target_text = "그렇군";
        assert!(
            assess_free_dialogue_anchor(&image, longer)
                .source_bound_fallback
                .is_none()
        );
    }

    #[test]
    fn compact_source_texts_select_only_complete_safe_evidence() {
        let mut image = GrayImage::from_pixel(220, 160, Luma([244]));
        for y in 56..84 {
            image.put_pixel(66, y, Luma([24]));
            image.put_pixel(70, y, Luma([24]));
        }
        for (source_text, target_text) in [("え〜", "에~"), ("ほう", "허")] {
            let mut candidate = input(&[]);
            candidate.source_text = source_text;
            candidate.target_text = target_text;
            let assessment = assess_free_dialogue_anchor(&image, candidate);
            let selected = assessment.selected_candidate.expect("safe candidate");
            assert!(selected.accepted, "{source_text}");
            assert!(selected.room.fits_target_length);
            assert!(selected.contrast.contrastable);
            assert!(selected.association.preserves_reading_order_and_attribution);
            assert!(selected.score >= MINIMUM_FREE_DIALOGUE_ANCHOR_SCORE);
            assert!(assessment.candidates.iter().all(|candidate| {
                candidate.score.is_finite()
                    && !candidate.rejection_reasons.contains(&"score_is_nan")
            }));
        }
    }

    #[test]
    fn structured_or_distant_patches_do_not_produce_an_anchor() {
        let mut image = GrayImage::from_pixel(220, 160, Luma([244]));
        for y in 0..160 {
            for x in 0..220 {
                if (x + y) % 4 < 2 {
                    image.put_pixel(x, y, Luma([24]));
                }
            }
        }
        let assessment = assess_free_dialogue_anchor(&image, input(&[]));
        assert!(assessment.selected_candidate.is_none());
        assert!(assessment.candidates.iter().any(|candidate| {
            candidate
                .rejection_reasons
                .contains(&"high_edge_density_or_character_face_outline_structure")
        }));

        let mut near_structure = GrayImage::from_pixel(420, 240, Luma([244]));
        let maximum_near_x = 180;
        for y in 0..240 {
            for x in 0..maximum_near_x {
                if (x + y) % 4 < 2 {
                    near_structure.put_pixel(x, y, Luma([24]));
                }
            }
        }
        let mut distant = input(&[]);
        distant.source_bounds = ElementBounds {
            x: 60.0,
            y: 50.0,
            width: 16.0,
            height: 40.0,
        };
        let assessment = assess_free_dialogue_anchor(&near_structure, distant);
        assert!(assessment.selected_candidate.is_none());
        assert!(assessment.candidates.iter().all(|candidate| {
            candidate.bounds.x + candidate.bounds.width <= 180.0 || !candidate.accepted
        }));
    }

    #[test]
    fn caption_sfx_ui_and_container_bound_roles_are_rejected() {
        let image = GrayImage::from_pixel(220, 160, Luma([244]));
        for role in [
            "dev.koharu.text.caption",
            "dev.koharu.text.sfx",
            "dev.koharu.text.ui",
        ] {
            let mut candidate = input(&[]);
            candidate.source_scene_role = role;
            let assessment = assess_free_dialogue_anchor(&image, candidate);
            assert!(assessment.selected_candidate.is_none(), "role {role}");
            assert!(assessment.source_bound_fallback.is_none(), "role {role}");
            assert_eq!(assessment.rejection_reasons, ["role_gate_rejected"]);
        }
        let mut container_bound = input(&[]);
        container_bound.has_finite_verified_container_relation = true;
        assert!(
            assess_free_dialogue_anchor(&image, container_bound)
                .selected_candidate
                .is_none()
        );
    }

    #[test]
    fn wrong_writing_modes_or_insufficient_target_room_are_rejected() {
        let image = GrayImage::from_pixel(220, 160, Luma([244]));
        let mut horizontal_source = input(&[]);
        horizontal_source.source_writing_mode = Some(WritingMode::Horizontal);
        assert!(
            assess_free_dialogue_anchor(&image, horizontal_source)
                .selected_candidate
                .is_none()
        );

        let mut long_target = input(&[]);
        long_target.target_text = "아주 긴 감탄 대사";
        let assessment = assess_free_dialogue_anchor(&image, long_target);
        assert!(assessment.selected_candidate.is_none());
        assert_eq!(
            assessment.rejection_reasons,
            ["target_length_exceeds_compact_anchor_capacity"]
        );

        let tiny = GrayImage::from_pixel(48, 64, Luma([244]));
        let mut no_room = input(&[]);
        no_room.source_bounds = ElementBounds {
            x: 16.0,
            y: 12.0,
            width: 16.0,
            height: 40.0,
        };
        let assessment = assess_free_dialogue_anchor(&tiny, no_room);
        assert!(assessment.selected_candidate.is_none());
        assert_eq!(
            assessment.rejection_reasons,
            ["no_adjacent_candidate_passed_every_raster_gate"]
        );
    }

    #[test]
    fn failed_required_uncontained_text_evidence_preserves_concrete_candidate_criteria() {
        let image = GrayImage::from_pixel(220, 160, Luma([24]));
        let assessment = assess_free_dialogue_anchor(&image, input(&[]));
        let evidence = failed_free_dialogue_anchor_evidence(&assessment).unwrap();
        assert!(evidence.role_gate.eligible_uncontained_source);
        assert!(evidence.writing_mode_gate.passed);
        assert_eq!(evidence.candidate_count, assessment.candidates.len());
        assert_eq!(evidence.best_rejected_candidates.len(), 3);
        assert!(
            evidence
                .rejection_reason_counts
                .contains_key("high_edge_density_or_character_face_outline_structure")
        );
    }
}
