use koharu_scene::{EntityId, Geometry, Origin, Revision, TextLayoutKind, Typography, WritingMode};
use serde::Serialize;
use serde_json::Value;

use crate::acceptance::{
    AcceptanceRejection, AcceptanceRejectionCode, ElementBounds, ProjectInspection,
    TargetLayoutRelation,
};
use crate::free_dialogue::{
    failed_free_dialogue_anchor_evidence, has_recorded_no_adjacent_candidate_outcome,
};
use crate::placement::is_actual_container_bound;

pub(crate) const CORRECTION_SCHEMA_VERSION: u32 = 4;
pub(crate) const REPAIR_PLAN_SCHEMA_VERSION: u32 = 13;

#[derive(Clone, Debug, Serialize)]
pub struct DeterministicRepairPlan {
    pub schema_version: u32,
    pub unresolved_failure_count: usize,
    pub blocking_failures: Vec<DeterministicRepairFailure>,
    pub terminal_diagnostic: Option<InfeasibleRequiredLayoutDiagnostic>,
}

#[derive(Clone, Debug, Serialize)]
pub struct InfeasibleRequiredLayoutDiagnostic {
    pub code: &'static str,
    pub element_id: EntityId,
    pub source_anchor_id: EntityId,
    pub source_anchor_bounds: ElementBounds,
    pub target_writing_mode: WritingMode,
    pub required_min_font_size_px: f64,
    pub required_contour_clearance_px: f64,
    pub required_cross_axis_px: f64,
    pub available_cross_axis_px: f64,
    pub immutable_constraints: Vec<&'static str>,
    pub allowed_operations: Vec<&'static str>,
    pub measured_candidates: Vec<InfeasibleLayoutCandidateMeasurement>,
    pub failed_adjacent_free_dialogue_anchor: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct InfeasibleLayoutCandidateMeasurement {
    pub candidate: &'static str,
    pub layout_bounds: ElementBounds,
    pub available_cross_axis_px: f64,
    pub rendered_font_size_px: Option<f64>,
    pub rendered_glyph_height_px: Option<f64>,
    pub minimum_clearance_px: Option<f64>,
    pub rejection_codes: Vec<AcceptanceRejectionCode>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeterministicRepairFailure {
    pub element_id: Option<EntityId>,
    pub logical_group_id: Option<EntityId>,
    pub primary_render_element_id: Option<EntityId>,
    pub member_ordinal_ids: Vec<RepairLogicalDialogueMember>,
    pub code: AcceptanceRejectionCode,
    pub expected: Value,
    pub actual: Value,
    pub allowed_repair_fields: Vec<RepairField>,
    pub next_action: Option<RepairNextAction>,
    pub required_evidence_revision: Revision,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RepairLogicalDialogueMember {
    pub ordinal: u32,
    pub element_id: EntityId,
    pub source_region_id: EntityId,
}

#[derive(Clone, Debug, Serialize)]
pub struct RepairNextAction {
    pub tool: &'static str,
    pub operation: &'static str,
    pub element: EntityId,
    pub required_evidence_revision: Revision,
    pub constraints: Vec<&'static str>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub(crate) struct RepairActionIdentity {
    pub operation: &'static str,
    pub element: EntityId,
}

impl RepairNextAction {
    pub(crate) fn identity(&self) -> RepairActionIdentity {
        RepairActionIdentity {
            operation: self.operation,
            element: self.element,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairField {
    SourceText,
    TranslationText,
    Typography,
    Layout,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RepairStopDiagnostic {
    pub schema_version: u32,
    pub reason: &'static str,
    pub previous_evidence_revision: Revision,
    pub reviewed_revision: Revision,
    pub previous_unresolved_failure_count: usize,
    pub current_unresolved_failure_count: usize,
    pub first_unresolved_failure: Option<DeterministicRepairFailure>,
    pub infeasible_required_layout: Option<InfeasibleRequiredLayoutDiagnostic>,
}

pub(crate) fn deterministic_repair_plan(
    rejections: &[AcceptanceRejection],
    required_evidence_revision: Revision,
    inspection: Option<&ProjectInspection>,
) -> DeterministicRepairPlan {
    let mut blocking_failures = rejections
        .iter()
        .filter(|rejection| !is_advisory_layout_evidence(rejection.code))
        .map(|rejection| {
            let logical_group = rejection.element_id.and_then(|element| {
                inspection?.project.pages.iter().find_map(|page| {
                    page.logical_dialogue_groups.iter().find(|group| {
                        group
                            .members
                            .iter()
                            .any(|member| member.element_id == element)
                    })
                })
            });
            let repair_element = logical_group
                .map(|group| group.primary_render_element_id)
                .or(rejection.element_id);
            let allowed_repair_fields = allowed_repair_fields(rejection.code);
            let (
                verified_ui_panel_anchor,
                verified_free_dialogue_anchor,
                source_bound_interjection_fallback,
            ) = repair_element
                .map(|element| verified_layout_routes(rejection, element, inspection))
                .unwrap_or_default();
            let next_action = match (rejection.code, repair_element) {
                (AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior, Some(element))
                    if verified_ui_panel_anchor
                        || verified_free_dialogue_anchor
                        || source_bound_interjection_fallback =>
                {
                    Some(text_layout_next_action(
                        element,
                        required_evidence_revision,
                        verified_ui_panel_anchor,
                        verified_free_dialogue_anchor,
                        source_bound_interjection_fallback,
                    ))
                }
                (AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior, Some(element)) => {
                    Some(RepairNextAction {
                        tool: "preview_increase_text_safe_padding",
                        operation: "increase_text_safe_padding",
                        element,
                        required_evidence_revision,
                        constraints: vec![
                            "every per-edge inset delta must be finite and greater than zero",
                            "candidate layout geometry may only shrink",
                            "the named target layout anchor and its contour remain authoritative; raw geometry is never accepted",
                            "preview must strictly improve and reach the configured text-safe clearance while independently passing every deterministic font, glyph, line-count, density, anchor-containment, clipping, page-overflow, overlap, and finite-layout constraint before commit",
                        ],
                    })
                }
                (AcceptanceRejectionCode::RenderedFontSizeBelowMinimum, Some(element))
                    if logical_group.is_some_and(|group| {
                        compact_translation_has_required_source_evidence(
                            group,
                            element,
                            inspection,
                        )
                    }) =>
                {
                    Some(compact_translation_next_action(
                        element,
                        required_evidence_revision,
                    ))
                }
                (_, Some(element))
                    if allowed_repair_fields
                        .iter()
                        .any(|field| matches!(field, RepairField::Typography | RepairField::Layout)) =>
                {
                    Some(text_layout_next_action(
                        element,
                        required_evidence_revision,
                        verified_ui_panel_anchor,
                        verified_free_dialogue_anchor,
                        source_bound_interjection_fallback,
                    ))
                }
                _ => None,
            };
            DeterministicRepairFailure {
                element_id: rejection.element_id,
                logical_group_id: logical_group.map(|group| group.group_id),
                primary_render_element_id: logical_group
                    .map(|group| group.primary_render_element_id),
                member_ordinal_ids: logical_group.map_or_else(Vec::new, |group| {
                    group
                        .members
                        .iter()
                        .map(|member| RepairLogicalDialogueMember {
                            ordinal: member.ordinal,
                            element_id: member.element_id,
                            source_region_id: member.source_region_id,
                        })
                        .collect()
                }),
                code: rejection.code,
                expected: rejection.expected.clone(),
                actual: rejection.actual.clone(),
                allowed_repair_fields,
                next_action,
                required_evidence_revision,
            }
        })
        .collect::<Vec<_>>();
    let terminal_diagnostic = blocking_failures
        .first()
        .and_then(|failure| infeasible_required_layout(failure, rejections, inspection));
    if let Some(diagnostic) = terminal_diagnostic.as_ref() {
        for failure in blocking_failures
            .iter_mut()
            .filter(|failure| failure.element_id == Some(diagnostic.element_id))
        {
            failure.next_action = None;
        }
    }
    DeterministicRepairPlan {
        schema_version: REPAIR_PLAN_SCHEMA_VERSION,
        unresolved_failure_count: blocking_failures.len(),
        blocking_failures,
        terminal_diagnostic,
    }
}

fn is_advisory_layout_evidence(code: AcceptanceRejectionCode) -> bool {
    matches!(
        code,
        AcceptanceRejectionCode::RenderedFontSizeBelowMinimum
            | AcceptanceRejectionCode::RenderedGlyphHeightBelowMinimum
            | AcceptanceRejectionCode::SourceRegionCoverageBelowMinimum
            | AcceptanceRejectionCode::TargetAnchorCoverageBelowMinimum
            | AcceptanceRejectionCode::RenderedTextOutsideSourceRegion
            | AcceptanceRejectionCode::RenderedTextOutsideTargetAnchor
            | AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior
    )
}

fn infeasible_required_layout(
    failure: &DeterministicRepairFailure,
    rejections: &[AcceptanceRejection],
    inspection: Option<&ProjectInspection>,
) -> Option<InfeasibleRequiredLayoutDiagnostic> {
    if failure.code != AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior {
        return None;
    }
    let inspection = inspection?;
    let element_id = failure.element_id?;
    let element = inspection
        .project
        .pages
        .iter()
        .flat_map(|page| &page.text_elements)
        .find(|element| element.id == element_id && element.required)?;
    if element.verified_ui_panel_anchor.is_some() || element.verified_free_dialogue_anchor.is_some()
    {
        return None;
    }
    let source_anchor = element.source_geometry.as_ref()?;
    let text_safe_region = element.text_safe_region.as_ref()?;
    let measured_region_id =
        serde_json::from_value::<EntityId>(failure.actual.get("region_id")?.clone()).ok()?;
    if is_actual_container_bound(element) || text_safe_region.id != measured_region_id {
        return None;
    }
    let target_anchor = failure.expected.get("target_layout_anchor")?;
    if target_anchor.get("kind")?.as_str()? != "source_region" {
        return None;
    }
    let target_writing_mode = match target_anchor.get("target_writing_mode")?.as_str()? {
        "Horizontal" | "horizontal" => WritingMode::Horizontal,
        "Vertical" | "vertical" => WritingMode::Vertical,
        _ => return None,
    };
    let thresholds = &inspection.configuration.quality_thresholds;
    let required_cross_axis_px =
        thresholds.min_rendered_font_size_px + thresholds.min_text_safe_padding_px * 2.0;
    let available_cross_axis_px = match target_writing_mode {
        WritingMode::Horizontal => source_anchor.bounds.height,
        WritingMode::Vertical => source_anchor.bounds.width,
    };
    if !required_cross_axis_px.is_finite() || !available_cross_axis_px.is_finite() {
        return None;
    }
    let clearance_boundary_bounds = ElementBounds {
        x: source_anchor.bounds.x + thresholds.min_text_safe_padding_px,
        y: source_anchor.bounds.y + thresholds.min_text_safe_padding_px,
        width: source_anchor.bounds.width - thresholds.min_text_safe_padding_px * 2.0,
        height: source_anchor.bounds.height - thresholds.min_text_safe_padding_px * 2.0,
    };
    let target_rejection_codes = rejections
        .iter()
        .filter(|rejection| {
            rejection.element_id == Some(element_id)
                || rejection.related_element_id == Some(element_id)
        })
        .map(|rejection| rejection.code)
        .collect::<Vec<_>>();
    let current_font_size = element.final_scene.font_size_px;
    let current_layout_is_source_bound = element
        .final_scene
        .layout_bounds
        .is_some_and(|bounds| bounds_cover_same_area(bounds, source_anchor.bounds));
    let padding_cannot_raise_undersized_font = current_layout_is_source_bound
        && current_font_size.is_some_and(|font| font < thresholds.min_rendered_font_size_px)
        && target_rejection_codes.contains(&AcceptanceRejectionCode::RenderedFontSizeBelowMinimum);
    let positively_required_uncontained_text_without_anchor = element
        .free_dialogue_anchor_assessment
        .as_ref()
        .is_some_and(|assessment| {
            assessment.role_gate.required
                && assessment.role_gate.eligible_uncontained_source
                && assessment.role_gate.classified_role == "required_uncontained_compact_text"
                && assessment.selected_candidate.is_none()
        });
    let source_bound_interjection_fallback = element
        .free_dialogue_anchor_assessment
        .as_ref()
        .is_some_and(|assessment| {
            assessment.selected_candidate.is_none()
                && has_recorded_no_adjacent_candidate_outcome(assessment)
                && assessment.source_bound_fallback.is_some()
                && assessment.role_gate.required
                && assessment.role_gate.eligible_uncontained_source
                && assessment.writing_mode_gate.passed
        });
    if source_bound_interjection_fallback {
        return None;
    }
    if available_cross_axis_px >= required_cross_axis_px
        && !padding_cannot_raise_undersized_font
        && !positively_required_uncontained_text_without_anchor
    {
        return None;
    }
    let current_minimum_clearance = failure
        .actual
        .get("minimum_clearance_px")
        .and_then(serde_json::Value::as_f64);
    Some(InfeasibleRequiredLayoutDiagnostic {
        code: "infeasible_required_layout",
        element_id,
        source_anchor_id: text_safe_region.id,
        source_anchor_bounds: source_anchor.bounds,
        target_writing_mode,
        required_min_font_size_px: thresholds.min_rendered_font_size_px,
        required_contour_clearance_px: thresholds.min_text_safe_padding_px,
        required_cross_axis_px,
        available_cross_axis_px,
        immutable_constraints: vec![
            "required content remains source-bound and cannot be skipped as difficult SFX",
            "the detected source anchor and text-safe contour are immutable",
            "minimum rendered font size and contour clearance thresholds remain unchanged",
        ],
        allowed_operations: if positively_required_uncontained_text_without_anchor {
            vec![
                "no mutation is authorized: source evidence classifies required dialogue and no target anchor passed every policy gate",
            ]
        } else {
            vec!["strictly inset all four layout edges within the source anchor"]
        },
        measured_candidates: vec![
            InfeasibleLayoutCandidateMeasurement {
                candidate: "current_render",
                layout_bounds: element
                    .final_scene
                    .layout_bounds
                    .unwrap_or(source_anchor.bounds),
                available_cross_axis_px,
                rendered_font_size_px: current_font_size,
                rendered_glyph_height_px: element
                    .final_scene
                    .glyph_bounds
                    .map(|bounds| bounds.height),
                minimum_clearance_px: current_minimum_clearance,
                rejection_codes: target_rejection_codes,
            },
            InfeasibleLayoutCandidateMeasurement {
                candidate: "maximum_required_clearance_interior",
                layout_bounds: clearance_boundary_bounds,
                available_cross_axis_px: match target_writing_mode {
                    WritingMode::Horizontal => clearance_boundary_bounds.height,
                    WritingMode::Vertical => clearance_boundary_bounds.width,
                },
                rendered_font_size_px: None,
                rendered_glyph_height_px: None,
                minimum_clearance_px: Some(thresholds.min_text_safe_padding_px),
                rejection_codes: vec![AcceptanceRejectionCode::RenderedFontSizeBelowMinimum],
            },
        ],
        failed_adjacent_free_dialogue_anchor: element
            .free_dialogue_anchor_assessment
            .as_ref()
            .and_then(failed_free_dialogue_anchor_evidence)
            .and_then(|evidence| serde_json::to_value(evidence).ok()),
    })
}

fn verified_layout_routes(
    rejection: &AcceptanceRejection,
    element: EntityId,
    inspection: Option<&ProjectInspection>,
) -> (bool, bool, bool) {
    let measured_anchor_kind = rejection
        .expected
        .pointer("/target_layout_anchor/kind")
        .or_else(|| rejection.actual.pointer("/target_layout_anchor/kind"))
        .and_then(Value::as_str);
    let inspected_element = inspection.and_then(|inspection| {
        inspection
            .project
            .pages
            .iter()
            .flat_map(|page| &page.text_elements)
            .find(|candidate| candidate.id == element && candidate.required)
    });
    (
        measured_anchor_kind == Some("ui_panel_text_safe_interior")
            || inspected_element
                .is_some_and(|candidate| candidate.verified_ui_panel_anchor.is_some()),
        measured_anchor_kind == Some("adjacent_free_dialogue_text_safe_anchor")
            || inspected_element
                .is_some_and(|candidate| candidate.verified_free_dialogue_anchor.is_some()),
        inspected_element.is_some_and(|candidate| {
            candidate.required
                && candidate.decorative_sfx.is_none()
                && candidate.verified_ui_panel_anchor.is_none()
                && candidate.verified_free_dialogue_anchor.is_none()
                && candidate
                    .free_dialogue_anchor_assessment
                    .as_ref()
                    .is_some_and(|assessment| {
                        assessment.selected_candidate.is_none()
                            && has_recorded_no_adjacent_candidate_outcome(assessment)
                            && assessment.source_bound_fallback.is_some()
                            && assessment.role_gate.required
                            && assessment.role_gate.eligible_uncontained_source
                            && assessment.writing_mode_gate.passed
                    })
        }),
    )
}

fn text_layout_next_action(
    element: EntityId,
    required_evidence_revision: Revision,
    verified_ui_panel_anchor: bool,
    verified_free_dialogue_anchor: bool,
    source_bound_interjection_fallback: bool,
) -> RepairNextAction {
    let mut constraints = vec![
        "inspect_page_evidence establishes bundled whole-page evidence before layout repair",
        "the failure's named page-level or visible-layout safety condition governs the repair",
        "raw bounds and replacement wording are not accepted; use only controlled preview_text_layout options",
        "preview must preserve semantic ownership, source identity, and any explicitly committed target relation",
        "preview must retain finite positive visible rendering and pass page clipping and translated-text overlap gates before commit_text_layout",
        "preview must be safe and strictly improve the blocking deterministic result before commit_text_layout",
    ];
    if verified_ui_panel_anchor {
        constraints.push(
            "use only controlled options for the retained required UI classification and its exact detector-backed panel/source association; native fallback, nearest-region selection, inferred bounds, free-text fallback, and SFX skip are forbidden",
        );
    } else if verified_free_dialogue_anchor {
        constraints.push(
            "use only controlled options for the accepted source-raster adjacent free-dialogue target; native fallback is forbidden while this verified candidate remains accepted",
        );
        constraints.push(
            "the exact target is the recorded text-safe region whose original-pixel boundary, room, contrast, reading-order, and attribution gates all passed",
        );
    } else if source_bound_interjection_fallback {
        constraints.push(
            "the source-bound native-vertical fallback is permitted only by a recorded no_adjacent_candidate_passed_every_raster_gate outcome plus the required compact no-container role, writing-mode, target-length, and immutable 12px/4px source-room gates",
        );
        constraints.push(
            "options JSON is exactly {\"source_bound_interjection_fallback\":\"native_vertical\"}; line_break_policy, max_lines, alignment, font_scale, and safe_padding_increase_px are forbidden because native defaults are host-owned",
        );
    }
    RepairNextAction {
        tool: "preview_text_layout",
        operation: if verified_ui_panel_anchor {
            "controlled_verified_ui_panel_target_layout"
        } else if verified_free_dialogue_anchor {
            "controlled_source_raster_verified_free_dialogue_target_layout"
        } else if source_bound_interjection_fallback {
            "controlled_source_bound_vertical_interjection_fallback"
        } else {
            "controlled_text_layout"
        },
        element,
        required_evidence_revision,
        constraints,
    }
}

fn bounds_cover_same_area(left: ElementBounds, right: ElementBounds) -> bool {
    const EPSILON: f64 = 0.001;
    (left.x - right.x).abs() <= EPSILON
        && (left.y - right.y).abs() <= EPSILON
        && (left.width - right.width).abs() <= EPSILON
        && (left.height - right.height).abs() <= EPSILON
}

fn compact_translation_has_required_source_evidence(
    group: &crate::acceptance::LogicalDialogueGroupInspection,
    primary_render_element: EntityId,
    inspection: Option<&ProjectInspection>,
) -> bool {
    let Some(inspection) = inspection else {
        return false;
    };
    if inspection.configuration.source_language != "ja-JP"
        || inspection.configuration.target_language != "ko-KR"
        || group.primary_render_element_id != primary_render_element
        || group.members.len() < 2
    {
        return false;
    }
    let Some(page) = inspection.project.pages.iter().find(|page| {
        page.logical_dialogue_groups
            .iter()
            .any(|candidate| candidate.group_id == group.group_id)
    }) else {
        return false;
    };
    let Some(primary) = page
        .text_elements
        .iter()
        .find(|element| element.id == primary_render_element)
    else {
        return false;
    };
    let verified_dialogue_target = primary.required
        && primary.source_writing_mode == Some(WritingMode::Vertical)
        && primary
            .text_safe_region
            .as_ref()
            .and_then(|region| region.association)
            .is_some_and(|association| {
                association.layout_relation == TargetLayoutRelation::FlowsIn
                    && association.source_inside_target_relation
            });
    let current_korean = primary.translation.as_ref().is_some_and(|translation| {
        translation.language.as_deref() == Some("ko-KR") && !translation.text.trim().is_empty()
    });
    verified_dialogue_target
        && current_korean
        && group.members.iter().all(|member| {
            page.text_elements
                .iter()
                .find(|element| element.id == member.element_id)
                .and_then(|element| element.source.as_ref())
                .is_some_and(|source| {
                    source.language.as_deref() == Some("ja-JP") && !source.text.trim().is_empty()
                })
        })
}

pub(crate) fn compact_translation_next_action(
    element: EntityId,
    required_evidence_revision: Revision,
) -> RepairNextAction {
    RepairNextAction {
        tool: "preview_compact_translation",
        operation: "layout_constrained_translation_compaction",
        element,
        required_evidence_revision,
        constraints: vec![
            "only the active logical dialogue group's primary render element and immutable ordered source members may be targeted",
            "all four fresh source/translation evidence digests for the required revision are mandatory",
            "candidate text must be nonempty ko-KR and contain strictly fewer visible grapheme units than the current translation",
            "raw geometry, typography, source/OCR edits, and general semantic mutation are not accepted",
            "preview must retain finite positive visible rendering and pass page clipping and translated-text overlap gates; font, glyph, anchor, and contour-clearance measurements remain diagnostic",
            "commit requires the valid fresh preview id; the next in-loop semantic review must compare the compact wording with every original group-member crop",
        ],
    }
}

pub(crate) fn is_long_dialogue_target_anchor_failure(failure: &DeterministicRepairFailure) -> bool {
    failure.code == AcceptanceRejectionCode::TargetAnchorCoverageBelowMinimum
        && failure.logical_group_id.is_some()
        && failure
            .expected
            .pointer("/target_layout_anchor/coverage_policy")
            .and_then(Value::as_str)
            == Some("configured_long_dialogue_floor")
}

pub(crate) fn is_long_dialogue_clearance_failure(failure: &DeterministicRepairFailure) -> bool {
    failure.code == AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior
        && failure.logical_group_id.is_some()
        && failure.member_ordinal_ids.len() >= 2
        && failure
            .expected
            .pointer("/target_layout_anchor/coverage_policy")
            .and_then(Value::as_str)
            == Some("configured_long_dialogue_floor")
}

fn allowed_repair_fields(code: AcceptanceRejectionCode) -> Vec<RepairField> {
    use AcceptanceRejectionCode as Code;
    use RepairField as Field;

    match code {
        Code::SourceTextMissing
        | Code::SourceTextEmpty
        | Code::SourceLanguageMissing
        | Code::SourceLanguageMismatch => vec![Field::SourceText],
        Code::TranslationMissing
        | Code::TranslationEmpty
        | Code::TranslationLanguageMissing
        | Code::TranslationLanguageMismatch => vec![Field::TranslationText],
        Code::DuplicateTranslationRegion => {
            vec![Field::TranslationText, Field::Typography, Field::Layout]
        }
        Code::TranslationNotRenderEligible
        | Code::TranslationGeometryNotVisible
        | Code::RenderedGlyphInkMissing
        | Code::UnsupportedLayout
        | Code::RenderedFontSizeMissing
        | Code::RenderedFontSizeBelowMinimum
        | Code::RenderedLineCountMissing
        | Code::RenderedLineCountUnreasonable
        | Code::RenderedGlyphHeightBelowMinimum
        | Code::SourceRegionCoverageBelowMinimum
        | Code::RenderedTextOutsideSourceRegion
        | Code::TargetAnchorCoverageBelowMinimum
        | Code::RenderedTextOutsideTargetAnchor
        | Code::RenderedTextOutsidePage
        | Code::TranslatedRegionOverlap => vec![Field::Typography, Field::Layout],
        Code::RenderedTextOutsideTextSafeInterior => Vec::new(),
        Code::PageNoDetectedSourceText
        | Code::InvalidDecorativeSfxDecision
        | Code::LogicalDialogueOwnershipConflict
        | Code::LogicalDialogueMemberMismatch
        | Code::LogicalDialogueNonPrimaryTranslation
        | Code::LogicalDialogueNonPrimaryVisible
        | Code::TypesettingFailed
        | Code::TranslationHidden
        | Code::TranslationZeroOpacity
        | Code::SourceRegionGeometryMissing
        | Code::TextSafeRegionMissing
        | Code::SourceRegionOverlap => Vec::new(),
    }
}

pub(crate) fn repair_stop_diagnostic(
    previous: &DeterministicRepairPlan,
    current: &DeterministicRepairPlan,
) -> Option<RepairStopDiagnostic> {
    let previous_failure = previous.blocking_failures.first()?;
    let previous_action = previous_failure.next_action.as_ref()?.identity();
    let current_failure = current.blocking_failures.first()?;
    if current_failure.next_action.as_ref()?.identity() != previous_action {
        return None;
    }
    Some(RepairStopDiagnostic {
        schema_version: REPAIR_PLAN_SCHEMA_VERSION,
        reason: "deterministic repair made no progress",
        previous_evidence_revision: previous_failure.required_evidence_revision,
        reviewed_revision: current
            .blocking_failures
            .first()
            .map_or(previous_failure.required_evidence_revision, |failure| {
                failure.required_evidence_revision
            }),
        previous_unresolved_failure_count: previous.unresolved_failure_count,
        current_unresolved_failure_count: current.unresolved_failure_count,
        first_unresolved_failure: Some(current_failure.clone()),
        infeasible_required_layout: None,
    })
}

pub(crate) fn infeasible_repair_stop_diagnostic(
    plan: &DeterministicRepairPlan,
) -> Option<RepairStopDiagnostic> {
    let diagnostic = plan.terminal_diagnostic.clone()?;
    let failure = plan.blocking_failures.first()?;
    Some(RepairStopDiagnostic {
        schema_version: REPAIR_PLAN_SCHEMA_VERSION,
        reason: "infeasible_required_layout",
        previous_evidence_revision: failure.required_evidence_revision,
        reviewed_revision: failure.required_evidence_revision,
        previous_unresolved_failure_count: plan.unresolved_failure_count,
        current_unresolved_failure_count: plan.unresolved_failure_count,
        first_unresolved_failure: Some(failure.clone()),
        infeasible_required_layout: Some(diagnostic),
    })
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RepairHistory {
    pub completed_review_attempts: u32,
    pub attempted_actions: Vec<RepairActionIdentity>,
    pub active_deterministic_plan: Option<DeterministicRepairPlan>,
    pub stop_diagnostic: Option<RepairStopDiagnostic>,
    pub correction_actions: Vec<CorrectionRecord>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CorrectionRecord {
    pub schema_version: u32,
    pub sequence: u32,
    pub review_attempt: u32,
    pub revision_before: Revision,
    pub revision_after: Revision,
    pub element_id: EntityId,
    pub content_id: EntityId,
    pub original_ocr: Option<RepairText>,
    pub evidence: RevisionEvidence,
    pub changed_fields: Vec<&'static str>,
    pub before: RepairElementState,
    pub after: RepairElementState,
    pub reason: String,
    pub agent_action: AgentAction,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum RevisionEvidence {
    DeterministicRejection {
        revision: Revision,
        review_attempt: u32,
    },
    ExternalReviewRejection {
        revision: Revision,
        review_attempt: u32,
    },
    ProjectInspection {
        revision: Revision,
    },
    RenderedPageInspection {
        revision: Revision,
        page_id: EntityId,
    },
    PageTranslationReview {
        revision: Revision,
        page_id: EntityId,
        dossier_digest: String,
    },
    RenderedPageDebugInspection {
        revision: Revision,
        page_id: EntityId,
        artifact_digest: String,
    },
    SourceEvidenceInspection {
        revision: Revision,
        page_id: EntityId,
        dossier_digest: String,
    },
    SourcePageDebugInspection {
        revision: Revision,
        page_id: EntityId,
        artifact_digest: String,
    },
    PageTranslationVisualEvidence {
        revision: Revision,
        page_id: EntityId,
        source_evidence_dossier_blake3: String,
        source_debug_artifact_blake3: String,
        dossier_blake3: String,
        debug_artifact_blake3: String,
    },
}

impl RevisionEvidence {
    pub(crate) fn revision(&self) -> Revision {
        match self {
            Self::DeterministicRejection { revision, .. }
            | Self::ExternalReviewRejection { revision, .. }
            | Self::ProjectInspection { revision }
            | Self::RenderedPageInspection { revision, .. }
            | Self::PageTranslationReview { revision, .. }
            | Self::RenderedPageDebugInspection { revision, .. }
            | Self::SourceEvidenceInspection { revision, .. }
            | Self::SourcePageDebugInspection { revision, .. }
            | Self::PageTranslationVisualEvidence { revision, .. } => *revision,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct AgentAction {
    pub actor: &'static str,
    pub configured_model: Option<String>,
    pub tool: &'static str,
    pub tool_call_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct RepairElementState {
    pub source: Option<RepairText>,
    pub translation: Option<RepairText>,
    pub typography: Option<Typography>,
    pub layout_kind: TextLayoutKind,
    pub authored_layout_geometry: Option<Geometry>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct RepairText {
    pub text: String,
    pub language: Option<String>,
    pub origin: Origin,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn legacy_geometry_thresholds_never_enter_deterministic_repair_routing() {
        let revision = Revision::new(7);
        let codes = [
            AcceptanceRejectionCode::RenderedFontSizeBelowMinimum,
            AcceptanceRejectionCode::RenderedGlyphHeightBelowMinimum,
            AcceptanceRejectionCode::SourceRegionCoverageBelowMinimum,
            AcceptanceRejectionCode::TargetAnchorCoverageBelowMinimum,
            AcceptanceRejectionCode::RenderedTextOutsideSourceRegion,
            AcceptanceRejectionCode::RenderedTextOutsideTargetAnchor,
            AcceptanceRejectionCode::RenderedTextOutsideTextSafeInterior,
        ];
        let rejections = codes.map(|code| AcceptanceRejection {
            code,
            page_id: EntityId::new(),
            element_id: Some(EntityId::new()),
            related_element_id: None,
            expected: json!({}),
            actual: json!({}),
        });

        let plan = deterministic_repair_plan(&rejections, revision, None);

        assert_eq!(plan.unresolved_failure_count, 0);
        assert!(plan.blocking_failures.is_empty());
        assert!(plan.terminal_diagnostic.is_none());
    }
}
