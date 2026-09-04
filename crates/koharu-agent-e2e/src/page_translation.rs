use std::{cmp::Ordering, io::Cursor, path::Path};

use anyhow::{Context as _, Result};
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use koharu_scene::{EntityId, Revision, Typography};
use serde::Serialize;

use crate::acceptance::{
    ElementBounds, ElementGeometry, LayoutAnchorMeasurement, LogicalDialogueGroupInspection,
    LogicalDialogueMembershipInspection, PageAcceptance, PageInspection, SemanticText,
    TextSafeContainment,
};
use crate::free_dialogue::{FreeDialogueAnchorAssessment, FreeDialogueAnchorDecision};
use crate::ui_panel::{DetectedPanelCandidate, UiPanelAnchorDecision};

pub(crate) const PAGE_TRANSLATION_DOSSIER_SCHEMA_VERSION: u32 = 9;
pub(crate) const PAGE_DEBUG_SCHEMA_VERSION: u32 = 6;
pub(crate) const SOURCE_EVIDENCE_DOSSIER_SCHEMA_VERSION: u32 = 6;
pub(crate) const PAGE_SOURCE_DEBUG_SCHEMA_VERSION: u32 = 5;

pub(crate) const ORIGINAL_PIXELS_AUTHORITY_CONTRACT: &str = "The original full-page and crop pixels are the authority for source meaning. OCR and current source text are fallible aids only and must not independently justify a source-text correction.";

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RenderedPageReference {
    pub media_type: &'static str,
    pub byte_length: usize,
    pub blake3: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PageTranslationDossier {
    pub schema_version: u32,
    pub scene_revision: Revision,
    pub page_id: EntityId,
    pub page_label: String,
    pub page_size_px: ElementBounds,
    pub reading_order_policy: &'static str,
    pub source_language: String,
    pub target_language: String,
    pub semantic_assessment: &'static str,
    pub page_context: PageDialogueContext,
    pub style_constraints: Vec<&'static str>,
    pub rendered_page: RenderedPageReference,
    pub logical_dialogue_groups: Vec<LogicalDialogueGroupInspection>,
    pub detected_panel_candidates: Vec<DetectedPanelCandidate>,
    pub elements: Vec<PageDialogueElement>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PageDialogueContext {
    pub element_count: usize,
    pub required_content_count: usize,
    pub skipped_difficult_sfx_count: usize,
    pub ordered_source_dialogue: Vec<String>,
    pub ordered_translation_dialogue: Vec<String>,
    pub continuity_review_prompts: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PageDialogueElement {
    pub ordinal: usize,
    pub element_id: EntityId,
    pub short_stable_id: String,
    pub source_region_id: Option<EntityId>,
    pub text_role: Option<String>,
    pub detected: bool,
    pub required: bool,
    pub review_state: &'static str,
    pub decorative_sfx: Option<crate::sfx::DecorativeSfxDecision>,
    pub original_ocr: Option<SemanticText>,
    pub current_source: Option<SemanticText>,
    pub current_translation: Option<SemanticText>,
    pub logical_dialogue_memberships: Vec<LogicalDialogueMembershipInspection>,
    pub verified_ui_panel_anchor: Option<UiPanelAnchorDecision>,
    pub verified_free_dialogue_anchor: Option<FreeDialogueAnchorDecision>,
    pub free_dialogue_anchor_assessment: Option<FreeDialogueAnchorAssessment>,
    pub visible_translation_owner: bool,
    pub geometry: DialogueGeometrySummary,
    pub typography: Option<Typography>,
    pub layout: DialogueLayoutMetrics,
    pub adjacency: DialogueAdjacency,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct DialogueGeometrySummary {
    pub source_region: Option<PolygonSummary>,
    pub text_safe_region: Option<PolygonSummary>,
    pub authored_layout_bounds: Option<ElementBounds>,
    pub rendered_text_bounds: Option<ElementBounds>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PolygonSummary {
    pub point_count: usize,
    pub bounds: ElementBounds,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct DialogueLayoutMetrics {
    pub layout_kind: String,
    pub rendered_layout_bounds: Option<ElementBounds>,
    pub rendered_font_size_px: Option<f64>,
    pub rendered_line_count: Option<usize>,
    pub rendered_lines: Vec<String>,
    pub renderer_diagnostics: Vec<String>,
    pub source_region_overflow_px: Option<f64>,
    pub target_layout_anchor: Option<LayoutAnchorMeasurement>,
    pub page_overflow_px: Option<f64>,
    pub text_safe_containment: Option<TextSafeContainment>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct DialogueAdjacency {
    pub previous_ordinal: Option<usize>,
    pub previous_element_id: Option<EntityId>,
    pub next_ordinal: Option<usize>,
    pub next_element_id: Option<EntityId>,
    pub nearest_element_ordinal: Option<usize>,
    pub nearest_element_id: Option<EntityId>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PageDebugArtifact {
    pub schema_version: u32,
    pub scene_revision: Revision,
    pub page_id: EntityId,
    pub path: String,
    pub media_type: &'static str,
    pub byte_length: usize,
    pub blake3: String,
    pub labels: Vec<PageDebugLabel>,
    pub legend: Vec<PageDebugLegend>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PageDebugLabel {
    pub ordinal: usize,
    pub element_id: EntityId,
    pub short_stable_id: String,
    pub label: String,
    pub review_state: &'static str,
    pub logical_group_id: Option<EntityId>,
    pub member_ordinal: Option<u32>,
    pub primary_render_element_id: Option<EntityId>,
    pub target_layout_anchor_kind: Option<String>,
    pub target_layout_anchor_region_id: Option<EntityId>,
    pub free_dialogue_anchor_decision: Option<FreeDialogueAnchorDecision>,
    pub free_dialogue_anchor_assessment: Option<FreeDialogueAnchorAssessment>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PageDebugLegend {
    pub name: &'static str,
    pub color_rgba: [u8; 4],
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SourceImageArtifact {
    pub path: String,
    pub media_type: String,
    pub byte_length: usize,
    pub blake3: String,
    pub width_px: u32,
    pub height_px: u32,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SourceEvidenceDossier {
    pub schema_version: u32,
    pub scene_revision: Revision,
    pub page_id: EntityId,
    pub page_label: String,
    pub reading_order_policy: &'static str,
    pub authority_contract: &'static str,
    pub full_page_original: SourceImageArtifact,
    pub logical_dialogue_groups: Vec<LogicalDialogueGroupInspection>,
    pub detected_panel_candidates: Vec<DetectedPanelCandidate>,
    pub elements: Vec<SourceEvidenceElement>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SourceEvidenceElement {
    pub ordinal: usize,
    pub element_id: EntityId,
    pub short_stable_id: String,
    pub source_debug_label: String,
    pub source_region_id: Option<EntityId>,
    pub text_role: Option<String>,
    pub required: bool,
    pub review_state: &'static str,
    pub decorative_sfx: Option<crate::sfx::DecorativeSfxDecision>,
    pub source_ocr: Option<SemanticText>,
    pub current_source: Option<SemanticText>,
    pub logical_dialogue_memberships: Vec<LogicalDialogueMembershipInspection>,
    pub verified_ui_panel_anchor: Option<UiPanelAnchorDecision>,
    pub verified_free_dialogue_anchor: Option<FreeDialogueAnchorDecision>,
    pub free_dialogue_anchor_assessment: Option<FreeDialogueAnchorAssessment>,
    pub ocr_confidence: Option<f32>,
    pub source_polygon: Option<ElementGeometry>,
    pub source_bounds: Option<ElementBounds>,
    pub crop_scope: &'static str,
    pub original_crop: SourceImageArtifact,
}

pub(crate) fn page_reading_order_ordinals(page: &PageInspection) -> Vec<(usize, EntityId)> {
    let mut ordered = page
        .text_elements
        .iter()
        .filter(|element| {
            element.detected || element.source.is_some() || element.translation.is_some()
        })
        .collect::<Vec<_>>();
    ordered.sort_by(reading_order_cmp);
    ordered
        .into_iter()
        .enumerate()
        .map(|(index, element)| (index + 1, element.id))
        .collect()
}

pub(crate) fn build_page_translation_dossier(
    revision: Revision,
    page: &PageInspection,
    page_acceptance: Option<&PageAcceptance>,
    source_language: &str,
    target_language: &str,
    original_ocr: impl Fn(EntityId, &Option<SemanticText>) -> Option<SemanticText>,
    rendered_page: RenderedPageReference,
) -> PageTranslationDossier {
    let ordinals = page_reading_order_ordinals(page);
    let ordered = ordinals
        .iter()
        .filter_map(|(_, id)| page.text_elements.iter().find(|element| element.id == *id))
        .collect::<Vec<_>>();
    let centers = ordered
        .iter()
        .map(|element| geometry_bounds(element).map(bounds_center))
        .collect::<Vec<_>>();
    let stable_ids =
        concise_stable_ids(&ordered.iter().map(|element| element.id).collect::<Vec<_>>());
    let elements = ordered
        .iter()
        .enumerate()
        .map(|(index, element)| {
            let measurements = page_acceptance.and_then(|page| {
                page.elements
                    .iter()
                    .find(|value| value.element_id == element.id)
                    .map(|value| &value.measurements)
            });
            let nearest = centers[index].and_then(|center| {
                centers
                    .iter()
                    .enumerate()
                    .filter_map(|(other_index, other)| {
                        (other_index != index).then_some((other_index, other.as_ref()?))
                    })
                    .min_by(|(first_index, first), (second_index, second)| {
                        squared_distance(center, **first)
                            .total_cmp(&squared_distance(center, **second))
                            .then_with(|| first_index.cmp(second_index))
                    })
                    .map(|(other_index, _)| other_index)
            });
            PageDialogueElement {
                ordinal: index + 1,
                element_id: element.id,
                short_stable_id: stable_ids[index].clone(),
                source_region_id: element.source_region_id,
                text_role: element.text_role.clone(),
                detected: element.detected,
                required: element.required,
                review_state: if element
                    .decorative_sfx
                    .as_ref()
                    .is_some_and(|value| value.is_skipped())
                {
                    "skipped_difficult_sfx"
                } else if element
                    .decorative_sfx
                    .as_ref()
                    .is_some_and(|value| value.is_translated())
                {
                    "translated_decorative_sfx"
                } else {
                    "required"
                },
                decorative_sfx: element.decorative_sfx.clone(),
                original_ocr: original_ocr(element.id, &element.source),
                current_source: element.source.clone(),
                current_translation: element.translation.clone(),
                logical_dialogue_memberships: element.logical_dialogue_memberships.clone(),
                verified_ui_panel_anchor: element.verified_ui_panel_anchor.clone(),
                verified_free_dialogue_anchor: element.verified_free_dialogue_anchor.clone(),
                free_dialogue_anchor_assessment: element.free_dialogue_anchor_assessment.clone(),
                visible_translation_owner: element
                    .decorative_sfx
                    .as_ref()
                    .is_none_or(crate::sfx::DecorativeSfxDecision::requires_translation)
                    && element
                        .logical_dialogue_memberships
                        .first()
                        .is_none_or(|membership| {
                            membership.primary_render_element_id == element.id
                        }),
                geometry: DialogueGeometrySummary {
                    source_region: element.source_geometry.as_ref().map(polygon_summary),
                    text_safe_region: element
                        .text_safe_region
                        .as_ref()
                        .map(|region| polygon_summary(&region.geometry)),
                    authored_layout_bounds: element
                        .authored_layout_geometry
                        .as_ref()
                        .map(|geometry| geometry.bounds),
                    rendered_text_bounds: element.final_scene.glyph_bounds,
                },
                typography: element.typography.clone(),
                layout: DialogueLayoutMetrics {
                    layout_kind: format!("{:?}", element.layout_kind).to_lowercase(),
                    rendered_layout_bounds: element.final_scene.layout_bounds,
                    rendered_font_size_px: element.final_scene.font_size_px,
                    rendered_line_count: element.final_scene.line_count,
                    rendered_lines: element.final_scene.rendered_lines.clone(),
                    renderer_diagnostics: element.final_scene.diagnostics.clone(),
                    source_region_overflow_px: measurements
                        .and_then(|value| value.source_region_overflow_px),
                    target_layout_anchor: measurements
                        .map(|value| value.target_layout_anchor.clone()),
                    page_overflow_px: measurements.and_then(|value| value.page_overflow_px),
                    text_safe_containment: measurements
                        .and_then(|value| value.text_safe_containment.clone()),
                },
                adjacency: DialogueAdjacency {
                    previous_ordinal: index.checked_sub(1).map(|value| value + 1),
                    previous_element_id: index.checked_sub(1).map(|value| ordered[value].id),
                    next_ordinal: (index + 1 < ordered.len()).then_some(index + 2),
                    next_element_id: ordered.get(index + 1).map(|value| value.id),
                    nearest_element_ordinal: nearest.map(|value| value + 1),
                    nearest_element_id: nearest.map(|value| ordered[value].id),
                },
            }
        })
        .collect::<Vec<_>>();

    PageTranslationDossier {
        schema_version: PAGE_TRANSLATION_DOSSIER_SCHEMA_VERSION,
        scene_revision: revision,
        page_id: page.id,
        page_label: page.label.clone(),
        page_size_px: ElementBounds {
            x: 0.0,
            y: 0.0,
            width: page.width,
            height: page.height,
        },
        reading_order_policy: "top_to_bottom_then_right_to_left_with_stable_id_tiebreak",
        source_language: source_language.to_owned(),
        target_language: target_language.to_owned(),
        semantic_assessment: "Evidence and deterministic ordering only; no OCR or translation semantic correctness is claimed.",
        page_context: PageDialogueContext {
            element_count: elements.len(),
            required_content_count: elements.iter().filter(|element| element.required).count(),
            skipped_difficult_sfx_count: elements
                .iter()
                .filter(|element| {
                    element
                        .decorative_sfx
                        .as_ref()
                        .is_some_and(|value| value.is_skipped())
                })
                .count(),
            ordered_source_dialogue: elements
                .iter()
                .filter(|element| element.required)
                .map(|element| {
                    element
                        .current_source
                        .as_ref()
                        .map_or_else(String::new, |text| text.text.clone())
                })
                .collect(),
            ordered_translation_dialogue: elements
                .iter()
                .filter(|element| element.required && element.visible_translation_owner)
                .map(|element| {
                    element
                        .current_translation
                        .as_ref()
                        .map_or_else(String::new, |text| text.text.clone())
                })
                .collect(),
            continuity_review_prompts: vec![
                "Check whether adjacent balloons continue one sentence or thought.",
                "Check speaker register, honorifics, pronouns, terminology, and tone across the whole page.",
                "Check whether parentheses add commentary absent from the source; remove them unless required.",
                "Treat wording and punctuation as semantic edits; treat visual line breaks as layout edits.",
            ],
        },
        style_constraints: vec![
            "Preserve source meaning and page-wide dialogue continuity.",
            "Use natural target-language dialogue with consistent register and terminology.",
            "Do not add explanatory parentheses or stage directions unsupported by the source.",
            "Do not infer semantic correctness from geometry or rendering acceptance.",
            "Use revise_page_translation for cross-balloon wording, then preview_text_layout for visual line breaks and bubble fit.",
        ],
        rendered_page,
        logical_dialogue_groups: page.logical_dialogue_groups.clone(),
        detected_panel_candidates: page.detected_panel_candidates.clone(),
        elements,
    }
}

pub(crate) fn build_source_evidence_dossier(
    root: &Path,
    revision: Revision,
    page: &PageInspection,
    original_bytes: &[u8],
    original_media_type: &str,
    source_ocr: impl Fn(EntityId, &Option<SemanticText>) -> Option<SemanticText>,
    ocr_confidence: impl Fn(Option<EntityId>) -> Option<f32>,
) -> Result<SourceEvidenceDossier> {
    let original = image::load_from_memory(original_bytes)
        .context("failed to decode original page for source evidence")?;
    let directory = root
        .join("source-evidence")
        .join(format!("revision-{revision}"))
        .join(format!("page-{}", page.id));
    std::fs::create_dir_all(&directory).with_context(|| {
        format!(
            "failed to create source evidence directory {}",
            directory.display()
        )
    })?;

    let original_digest = blake3::hash(original_bytes).to_hex().to_string();
    let original_path = directory.join(format!(
        "original-{}.{}",
        &original_digest[..12],
        media_extension(original_media_type)
    ));
    std::fs::write(&original_path, original_bytes).with_context(|| {
        format!(
            "failed to write original source artifact {}",
            original_path.display()
        )
    })?;
    let full_page_original = SourceImageArtifact {
        path: original_path.to_string_lossy().into_owned(),
        media_type: original_media_type.to_owned(),
        byte_length: original_bytes.len(),
        blake3: original_digest,
        width_px: original.width(),
        height_px: original.height(),
    };

    let ordinals = page_reading_order_ordinals(page);
    let stable_ids = concise_stable_ids(&ordinals.iter().map(|(_, id)| *id).collect::<Vec<_>>());
    let mut elements = Vec::with_capacity(ordinals.len());
    for ((ordinal, element_id), short_stable_id) in ordinals.iter().zip(stable_ids) {
        let element = page
            .text_elements
            .iter()
            .find(|element| element.id == *element_id)
            .context("source evidence element is missing from page inspection")?;
        let source_bounds = element
            .source_geometry
            .as_ref()
            .map(|geometry| geometry.bounds);
        let (crop_bounds, crop_scope) = source_bounds
            .map(|bounds| (bounds, "source_region_bbox"))
            .unwrap_or_else(|| {
                (
                    geometry_bounds(element).unwrap_or(ElementBounds {
                        x: 0.0,
                        y: 0.0,
                        width: page.width,
                        height: page.height,
                    }),
                    "fallback_context_bbox_no_source_region",
                )
            });
        let (x, y, width, height) = pixel_crop_bounds(
            crop_bounds,
            page.width,
            page.height,
            original.width(),
            original.height(),
        );
        let crop = original.crop_imm(x, y, width, height);
        let mut crop_bytes = Cursor::new(Vec::new());
        crop.write_to(&mut crop_bytes, ImageFormat::Png)?;
        let crop_bytes = crop_bytes.into_inner();
        let crop_digest = blake3::hash(&crop_bytes).to_hex().to_string();
        let crop_path = directory.join(format!(
            "ordinal-{ordinal}-{short_stable_id}-{}.png",
            &crop_digest[..12]
        ));
        std::fs::write(&crop_path, &crop_bytes)
            .with_context(|| format!("failed to write source crop {}", crop_path.display()))?;
        elements.push(SourceEvidenceElement {
            ordinal: *ordinal,
            element_id: *element_id,
            source_debug_label: debug_label(
                *ordinal,
                &short_stable_id,
                element.logical_dialogue_memberships.first(),
                element
                    .decorative_sfx
                    .as_ref()
                    .is_some_and(|value| value.is_skipped()),
            ),
            short_stable_id,
            source_region_id: element.source_region_id,
            text_role: element.text_role.clone(),
            required: element.required,
            review_state: if element
                .decorative_sfx
                .as_ref()
                .is_some_and(|value| value.is_skipped())
            {
                "skipped_difficult_sfx"
            } else if element
                .decorative_sfx
                .as_ref()
                .is_some_and(|value| value.is_translated())
            {
                "translated_decorative_sfx"
            } else {
                "required"
            },
            decorative_sfx: element.decorative_sfx.clone(),
            source_ocr: source_ocr(*element_id, &element.source),
            current_source: element.source.clone(),
            logical_dialogue_memberships: element.logical_dialogue_memberships.clone(),
            verified_ui_panel_anchor: element.verified_ui_panel_anchor.clone(),
            verified_free_dialogue_anchor: element.verified_free_dialogue_anchor.clone(),
            free_dialogue_anchor_assessment: element.free_dialogue_anchor_assessment.clone(),
            ocr_confidence: ocr_confidence(element.source_region_id),
            source_polygon: element.source_geometry.clone(),
            source_bounds,
            crop_scope,
            original_crop: SourceImageArtifact {
                path: crop_path.to_string_lossy().into_owned(),
                media_type: "image/png".to_owned(),
                byte_length: crop_bytes.len(),
                blake3: crop_digest,
                width_px: width,
                height_px: height,
            },
        });
    }

    Ok(SourceEvidenceDossier {
        schema_version: SOURCE_EVIDENCE_DOSSIER_SCHEMA_VERSION,
        scene_revision: revision,
        page_id: page.id,
        page_label: page.label.clone(),
        reading_order_policy: "top_to_bottom_then_right_to_left_with_stable_id_tiebreak",
        authority_contract: ORIGINAL_PIXELS_AUTHORITY_CONTRACT,
        full_page_original,
        logical_dialogue_groups: page.logical_dialogue_groups.clone(),
        detected_panel_candidates: page.detected_panel_candidates.clone(),
        elements,
    })
}

pub(crate) fn render_page_debug_overlay(
    rendered_preview: &[u8],
    page: &PageInspection,
    ordinals: &[(usize, EntityId)],
) -> Result<(Vec<u8>, Vec<PageDebugLabel>)> {
    let mut image = image::load_from_memory(rendered_preview)
        .context("failed to decode rendered page preview for debug overlay")?
        .to_rgba8();
    let scale_x = f64::from(image.width()) / page.width;
    let scale_y = f64::from(image.height()) / page.height;
    let stable_ids = concise_stable_ids(&ordinals.iter().map(|(_, id)| *id).collect::<Vec<_>>());
    let mut labels = Vec::new();
    for panel in &page.detected_panel_candidates {
        draw_polygon(
            &mut image,
            &panel.geometry,
            scale_x,
            scale_y,
            Rgba([245, 158, 11, 255]),
        );
    }
    for ((ordinal, id), short) in ordinals.iter().zip(stable_ids) {
        let Some(element) = page.text_elements.iter().find(|element| element.id == *id) else {
            continue;
        };
        if let Some(geometry) = &element.source_geometry {
            draw_polygon(
                &mut image,
                geometry,
                scale_x,
                scale_y,
                Rgba([239, 68, 68, 255]),
            );
        }
        if let Some(bounds) = element.final_scene.layout_bounds {
            draw_bounds(
                &mut image,
                bounds,
                scale_x,
                scale_y,
                Rgba([59, 130, 246, 255]),
            );
        }
        if let Some(bounds) = element.final_scene.glyph_bounds {
            draw_bounds(
                &mut image,
                bounds,
                scale_x,
                scale_y,
                Rgba([34, 197, 94, 255]),
            );
        }
        if let Some(anchor) = &element.verified_ui_panel_anchor {
            draw_polygon(
                &mut image,
                &anchor.panel.geometry,
                scale_x,
                scale_y,
                Rgba([168, 85, 247, 255]),
            );
        }
        if let Some(anchor) = &element.verified_free_dialogue_anchor {
            draw_bounds(
                &mut image,
                anchor.candidate_bounds,
                scale_x,
                scale_y,
                Rgba([6, 182, 212, 255]),
            );
        }
        let anchor = geometry_bounds(element).unwrap_or_default();
        let membership = element.logical_dialogue_memberships.first();
        let label = debug_label(
            *ordinal,
            &short,
            membership,
            element
                .decorative_sfx
                .as_ref()
                .is_some_and(|value| value.is_skipped()),
        );
        draw_label(
            &mut image,
            (anchor.x * scale_x).round() as i32,
            (anchor.y * scale_y).round() as i32,
            &label,
        );
        labels.push(PageDebugLabel {
            ordinal: *ordinal,
            element_id: *id,
            short_stable_id: short,
            label,
            review_state: if element
                .decorative_sfx
                .as_ref()
                .is_some_and(|value| value.is_skipped())
            {
                "skipped_difficult_sfx"
            } else if element
                .decorative_sfx
                .as_ref()
                .is_some_and(|value| value.is_translated())
            {
                "translated_decorative_sfx"
            } else {
                "required"
            },
            logical_group_id: membership.map(|membership| membership.group_id),
            member_ordinal: membership.map(|membership| membership.member_ordinal),
            primary_render_element_id: membership
                .map(|membership| membership.primary_render_element_id),
            target_layout_anchor_kind: element
                .verified_ui_panel_anchor
                .as_ref()
                .map(|_| "ui_panel_text_safe_interior".to_owned()),
            target_layout_anchor_region_id: element
                .verified_ui_panel_anchor
                .as_ref()
                .map(|anchor| anchor.panel.region_id),
            free_dialogue_anchor_decision: element.verified_free_dialogue_anchor.clone(),
            free_dialogue_anchor_assessment: element.free_dialogue_anchor_assessment.clone(),
        });
        if let Some(label) = labels.last_mut()
            && let Some(anchor) = &element.verified_free_dialogue_anchor
        {
            label.target_layout_anchor_kind =
                Some("adjacent_free_dialogue_text_safe_anchor".to_owned());
            label.target_layout_anchor_region_id = Some(anchor.target_region_id);
        }
    }
    draw_legend(&mut image);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image).write_to(&mut bytes, ImageFormat::Png)?;
    Ok((bytes.into_inner(), labels))
}

pub(crate) fn render_page_source_debug_overlay(
    original_page: &[u8],
    page: &PageInspection,
    ordinals: &[(usize, EntityId)],
) -> Result<(Vec<u8>, Vec<PageDebugLabel>)> {
    let mut image = image::load_from_memory(original_page)
        .context("failed to decode original page for source debug overlay")?
        .to_rgba8();
    let scale_x = f64::from(image.width()) / page.width;
    let scale_y = f64::from(image.height()) / page.height;
    let stable_ids = concise_stable_ids(&ordinals.iter().map(|(_, id)| *id).collect::<Vec<_>>());
    let mut labels = Vec::with_capacity(ordinals.len());
    for panel in &page.detected_panel_candidates {
        draw_polygon(
            &mut image,
            &panel.geometry,
            scale_x,
            scale_y,
            Rgba([245, 158, 11, 255]),
        );
    }
    for ((ordinal, id), short) in ordinals.iter().zip(stable_ids) {
        let Some(element) = page.text_elements.iter().find(|element| element.id == *id) else {
            continue;
        };
        if let Some(geometry) = &element.source_geometry {
            draw_polygon(
                &mut image,
                geometry,
                scale_x,
                scale_y,
                Rgba([239, 68, 68, 255]),
            );
        }
        if let Some(anchor) = &element.verified_ui_panel_anchor {
            draw_polygon(
                &mut image,
                &anchor.panel.geometry,
                scale_x,
                scale_y,
                Rgba([168, 85, 247, 255]),
            );
        }
        if let Some(anchor) = &element.verified_free_dialogue_anchor {
            draw_bounds(
                &mut image,
                anchor.candidate_bounds,
                scale_x,
                scale_y,
                Rgba([6, 182, 212, 255]),
            );
        }
        let anchor = element
            .source_geometry
            .as_ref()
            .map(|geometry| geometry.bounds)
            .or_else(|| geometry_bounds(element))
            .unwrap_or_default();
        let membership = element.logical_dialogue_memberships.first();
        let label = debug_label(
            *ordinal,
            &short,
            membership,
            element
                .decorative_sfx
                .as_ref()
                .is_some_and(|value| value.is_skipped()),
        );
        draw_label(
            &mut image,
            (anchor.x * scale_x).round() as i32,
            (anchor.y * scale_y).round() as i32,
            &label,
        );
        labels.push(PageDebugLabel {
            ordinal: *ordinal,
            element_id: *id,
            short_stable_id: short,
            label,
            review_state: if element
                .decorative_sfx
                .as_ref()
                .is_some_and(|value| value.is_skipped())
            {
                "skipped_difficult_sfx"
            } else if element
                .decorative_sfx
                .as_ref()
                .is_some_and(|value| value.is_translated())
            {
                "translated_decorative_sfx"
            } else {
                "required"
            },
            logical_group_id: membership.map(|membership| membership.group_id),
            member_ordinal: membership.map(|membership| membership.member_ordinal),
            primary_render_element_id: membership
                .map(|membership| membership.primary_render_element_id),
            target_layout_anchor_kind: element
                .verified_ui_panel_anchor
                .as_ref()
                .map(|_| "ui_panel_text_safe_interior".to_owned()),
            target_layout_anchor_region_id: element
                .verified_ui_panel_anchor
                .as_ref()
                .map(|anchor| anchor.panel.region_id),
            free_dialogue_anchor_decision: element.verified_free_dialogue_anchor.clone(),
            free_dialogue_anchor_assessment: element.free_dialogue_anchor_assessment.clone(),
        });
        if let Some(label) = labels.last_mut()
            && let Some(anchor) = &element.verified_free_dialogue_anchor
        {
            label.target_layout_anchor_kind =
                Some("adjacent_free_dialogue_text_safe_anchor".to_owned());
            label.target_layout_anchor_region_id = Some(anchor.target_region_id);
        }
    }
    draw_source_legend(&mut image);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image).write_to(&mut bytes, ImageFormat::Png)?;
    Ok((bytes.into_inner(), labels))
}

fn debug_label(
    ordinal: usize,
    short_element_id: &str,
    membership: Option<&LogicalDialogueMembershipInspection>,
    skipped_difficult_sfx: bool,
) -> String {
    let base = format!("{ordinal}:{short_element_id}").to_uppercase();
    if skipped_difficult_sfx {
        return format!("{base}/SFX-SKIP");
    }
    membership.map_or(base.clone(), |membership| {
        let normalized = membership.group_id.to_string().replace('-', "");
        let group = &normalized[normalized.len().saturating_sub(6)..];
        format!(
            "{base}/G{}.{}/P:{}",
            group.to_uppercase(),
            membership.member_ordinal,
            membership.primary_render_element_id
        )
    })
}

pub(crate) fn write_page_debug_artifact(
    root: &Path,
    revision: Revision,
    page: EntityId,
    bytes: &[u8],
    labels: Vec<PageDebugLabel>,
) -> Result<PageDebugArtifact> {
    let digest = blake3::hash(bytes).to_hex().to_string();
    let directory = root.join("debug").join(format!("revision-{revision}"));
    std::fs::create_dir_all(&directory).with_context(|| {
        format!(
            "failed to create page debug directory {}",
            directory.display()
        )
    })?;
    let path = directory.join(format!("page-{page}-{}.png", &digest[..12]));
    std::fs::write(&path, bytes)
        .with_context(|| format!("failed to write page debug overlay {}", path.display()))?;
    Ok(PageDebugArtifact {
        schema_version: PAGE_DEBUG_SCHEMA_VERSION,
        scene_revision: revision,
        page_id: page,
        path: path.to_string_lossy().into_owned(),
        media_type: "image/png",
        byte_length: bytes.len(),
        blake3: digest,
        labels,
        legend: vec![
            PageDebugLegend {
                name: "source_region_polygon_or_bbox",
                color_rgba: [239, 68, 68, 255],
            },
            PageDebugLegend {
                name: "layout_bounds",
                color_rgba: [59, 130, 246, 255],
            },
            PageDebugLegend {
                name: "rendered_text_bounds",
                color_rgba: [34, 197, 94, 255],
            },
            PageDebugLegend {
                name: "detected_panel_candidate",
                color_rgba: [245, 158, 11, 255],
            },
            PageDebugLegend {
                name: "verified_ui_panel_text_safe_anchor",
                color_rgba: [168, 85, 247, 255],
            },
            PageDebugLegend {
                name: "source_raster_verified_adjacent_free_dialogue_anchor",
                color_rgba: [6, 182, 212, 255],
            },
        ],
    })
}

pub(crate) fn write_page_source_debug_artifact(
    root: &Path,
    revision: Revision,
    page: EntityId,
    bytes: &[u8],
    labels: Vec<PageDebugLabel>,
) -> Result<PageDebugArtifact> {
    let digest = blake3::hash(bytes).to_hex().to_string();
    let directory = root
        .join("source-debug")
        .join(format!("revision-{revision}"));
    std::fs::create_dir_all(&directory).with_context(|| {
        format!(
            "failed to create page source debug directory {}",
            directory.display()
        )
    })?;
    let path = directory.join(format!("page-{page}-{}.png", &digest[..12]));
    std::fs::write(&path, bytes).with_context(|| {
        format!(
            "failed to write page source debug overlay {}",
            path.display()
        )
    })?;
    Ok(PageDebugArtifact {
        schema_version: PAGE_SOURCE_DEBUG_SCHEMA_VERSION,
        scene_revision: revision,
        page_id: page,
        path: path.to_string_lossy().into_owned(),
        media_type: "image/png",
        byte_length: bytes.len(),
        blake3: digest,
        labels,
        legend: vec![
            PageDebugLegend {
                name: "original_source_region_polygon_or_bbox",
                color_rgba: [239, 68, 68, 255],
            },
            PageDebugLegend {
                name: "detected_panel_candidate",
                color_rgba: [245, 158, 11, 255],
            },
            PageDebugLegend {
                name: "verified_ui_panel_text_safe_anchor",
                color_rgba: [168, 85, 247, 255],
            },
            PageDebugLegend {
                name: "source_raster_verified_adjacent_free_dialogue_anchor",
                color_rgba: [6, 182, 212, 255],
            },
        ],
    })
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

fn pixel_crop_bounds(
    bounds: ElementBounds,
    page_width: f64,
    page_height: f64,
    image_width: u32,
    image_height: u32,
) -> (u32, u32, u32, u32) {
    if page_width <= 0.0
        || page_height <= 0.0
        || !page_width.is_finite()
        || !page_height.is_finite()
        || image_width == 0
        || image_height == 0
    {
        return (0, 0, image_width.max(1), image_height.max(1));
    }
    let sx = f64::from(image_width) / page_width;
    let sy = f64::from(image_height) / page_height;
    let left = (bounds.x * sx)
        .floor()
        .clamp(0.0, f64::from(image_width - 1)) as u32;
    let top = (bounds.y * sy)
        .floor()
        .clamp(0.0, f64::from(image_height - 1)) as u32;
    let right = ((bounds.x + bounds.width) * sx)
        .ceil()
        .clamp(f64::from(left + 1), f64::from(image_width)) as u32;
    let bottom = ((bounds.y + bounds.height) * sy)
        .ceil()
        .clamp(f64::from(top + 1), f64::from(image_height)) as u32;
    (left, top, right - left, bottom - top)
}

fn reading_order_cmp(
    first: &&crate::acceptance::TextElementInspection,
    second: &&crate::acceptance::TextElementInspection,
) -> Ordering {
    match (geometry_bounds(first), geometry_bounds(second)) {
        (Some(first_bounds), Some(second_bounds)) => first_bounds
            .y
            .total_cmp(&second_bounds.y)
            .then_with(|| second_bounds.x.total_cmp(&first_bounds.x))
            .then_with(|| first.id.to_string().cmp(&second.id.to_string())),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => first.id.to_string().cmp(&second.id.to_string()),
    }
}

fn geometry_bounds(element: &crate::acceptance::TextElementInspection) -> Option<ElementBounds> {
    element
        .source_geometry
        .as_ref()
        .map(|geometry| geometry.bounds)
        .or_else(|| {
            element
                .text_safe_region
                .as_ref()
                .map(|region| region.geometry.bounds)
        })
        .or_else(|| {
            element
                .authored_layout_geometry
                .as_ref()
                .map(|geometry| geometry.bounds)
        })
        .or(element.final_scene.layout_bounds)
        .or(element.final_scene.glyph_bounds)
}

fn polygon_summary(geometry: &ElementGeometry) -> PolygonSummary {
    PolygonSummary {
        point_count: geometry.points.len(),
        bounds: geometry.bounds,
    }
}

fn bounds_center(bounds: ElementBounds) -> (f64, f64) {
    (
        bounds.x + bounds.width / 2.0,
        bounds.y + bounds.height / 2.0,
    )
}

fn squared_distance(first: (f64, f64), second: (f64, f64)) -> f64 {
    (first.0 - second.0).powi(2) + (first.1 - second.1).powi(2)
}

fn concise_stable_ids(ids: &[EntityId]) -> Vec<String> {
    let normalized = ids
        .iter()
        .map(|id| id.to_string().replace('-', ""))
        .collect::<Vec<_>>();
    normalized
        .iter()
        .enumerate()
        .map(|(index, value)| {
            (6..=value.len())
                .find_map(|length| {
                    let suffix = &value[value.len() - length..];
                    normalized
                        .iter()
                        .enumerate()
                        .all(|(other_index, other)| {
                            other_index == index || !other.ends_with(suffix)
                        })
                        .then(|| suffix.to_owned())
                })
                .expect("distinct entity IDs must have a unique full-length suffix")
        })
        .collect()
}

fn draw_polygon(
    image: &mut RgbaImage,
    geometry: &ElementGeometry,
    sx: f64,
    sy: f64,
    color: Rgba<u8>,
) {
    if geometry.points.len() < 2 {
        draw_bounds(image, geometry.bounds, sx, sy, color);
        return;
    }
    for (from, to) in geometry
        .points
        .iter()
        .zip(geometry.points.iter().cycle().skip(1))
        .take(geometry.points.len())
    {
        draw_line(
            image,
            (from.x * sx).round() as i32,
            (from.y * sy).round() as i32,
            (to.x * sx).round() as i32,
            (to.y * sy).round() as i32,
            color,
        );
    }
}

fn draw_bounds(image: &mut RgbaImage, bounds: ElementBounds, sx: f64, sy: f64, color: Rgba<u8>) {
    let (left, top) = (
        (bounds.x * sx).round() as i32,
        (bounds.y * sy).round() as i32,
    );
    let (right, bottom) = (
        ((bounds.x + bounds.width) * sx).round() as i32,
        ((bounds.y + bounds.height) * sy).round() as i32,
    );
    draw_line(image, left, top, right, top, color);
    draw_line(image, right, top, right, bottom, color);
    draw_line(image, right, bottom, left, bottom, color);
    draw_line(image, left, bottom, left, top, color);
}

fn draw_line(image: &mut RgbaImage, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: Rgba<u8>) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        put_pixel_checked(image, x0, y0, color);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let doubled = 2 * error;
        if doubled >= dy {
            error += dy;
            x0 += sx;
        }
        if doubled <= dx {
            error += dx;
            y0 += sy;
        }
    }
}

fn draw_label(image: &mut RgbaImage, x: i32, y: i32, text: &str) {
    let width = text.chars().count() as i32 * 4 + 3;
    for py in y.max(0)..(y + 8).min(image.height() as i32) {
        for px in x.max(0)..(x + width).min(image.width() as i32) {
            image.put_pixel(px as u32, py as u32, Rgba([17, 24, 39, 220]));
        }
    }
    for (index, character) in text.chars().enumerate() {
        draw_glyph(image, x + 2 + index as i32 * 4, y + 2, character);
    }
}

fn draw_legend(image: &mut RgbaImage) {
    for (index, (name, color)) in [
        ("SRC", Rgba([239, 68, 68, 255])),
        ("LAY", Rgba([59, 130, 246, 255])),
        ("TXT", Rgba([34, 197, 94, 255])),
        ("PNL", Rgba([245, 158, 11, 255])),
        ("UIA", Rgba([168, 85, 247, 255])),
    ]
    .iter()
    .enumerate()
    {
        let y = 4 + index as i32 * 10;
        for py in y..y + 7 {
            for px in 4..8 {
                put_pixel_checked(image, px, py, *color);
            }
        }
        draw_label(image, 10, y, name);
    }
}

fn draw_source_legend(image: &mut RgbaImage) {
    for (index, (name, color)) in [
        ("SRC", Rgba([239, 68, 68, 255])),
        ("PNL", Rgba([245, 158, 11, 255])),
        ("UIA", Rgba([168, 85, 247, 255])),
    ]
    .iter()
    .enumerate()
    {
        let y = 4 + index as i32 * 10;
        for py in y..y + 7 {
            for px in 4..8 {
                put_pixel_checked(image, px, py, *color);
            }
        }
        draw_label(image, 10, y, name);
    }
}

fn draw_glyph(image: &mut RgbaImage, x: i32, y: i32, character: char) {
    for (row, bits) in glyph(character.to_ascii_uppercase())
        .into_iter()
        .enumerate()
    {
        for column in 0..3 {
            if bits & (1 << (2 - column)) != 0 {
                put_pixel_checked(
                    image,
                    x + column,
                    y + row as i32,
                    Rgba([255, 255, 255, 255]),
                );
            }
        }
    }
}

fn put_pixel_checked(image: &mut RgbaImage, x: i32, y: i32, color: Rgba<u8>) {
    if x >= 0 && y >= 0 && x < image.width() as i32 && y < image.height() as i32 {
        image.put_pixel(x as u32, y as u32, color);
    }
}

fn glyph(c: char) -> [u8; 5] {
    match c {
        '0' => [7, 5, 5, 5, 7],
        '1' => [2, 6, 2, 2, 7],
        '2' => [7, 1, 7, 4, 7],
        '3' => [7, 1, 7, 1, 7],
        '4' => [5, 5, 7, 1, 1],
        '5' => [7, 4, 7, 1, 7],
        '6' => [7, 4, 7, 5, 7],
        '7' => [7, 1, 2, 2, 2],
        '8' => [7, 5, 7, 5, 7],
        '9' => [7, 5, 7, 1, 7],
        'A' => [2, 5, 7, 5, 5],
        'B' => [6, 5, 6, 5, 6],
        'C' => [7, 4, 4, 4, 7],
        'D' => [6, 5, 5, 5, 6],
        'E' => [7, 4, 6, 4, 7],
        'F' => [7, 4, 6, 4, 4],
        'L' => [4, 4, 4, 4, 7],
        'R' => [6, 5, 6, 5, 5],
        'S' => [7, 4, 7, 1, 7],
        'T' => [7, 2, 2, 2, 2],
        'X' => [5, 5, 2, 5, 5],
        'Y' => [5, 5, 2, 2, 2],
        ':' => [0, 2, 0, 2, 0],
        '-' => [0, 0, 7, 0, 0],
        _ => [0; 5],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::{
        ElementPoint, ElementVisibility, FinalSceneElement, TextElementInspection, TextSafeRegion,
    };
    use koharu_scene::TextLayoutKind;

    fn mark_first_as_skipped_difficult_sfx(page: &mut PageInspection) {
        let page_id = page.id;
        let element = &mut page.text_elements[0];
        element.required = false;
        element.text_role = Some(crate::sfx::SKIPPED_DIFFICULT_SFX_ROLE.to_owned());
        element.translation = None;
        element.visibility.local_visible = false;
        element.visibility.effective_visible = false;
        element.final_scene.visible = false;
        element.typography = Some(Typography {
            origin: koharu_scene::Origin::User,
            preferred_font: None,
            font_weight: None,
            font_style: None,
            size: Some(18.0),
            auto_fit: true,
            color: Some([0, 0, 0, 255]),
            stroke_color: None,
            stroke_width: None,
            alignment: None,
            writing_mode: None,
            extensions: Default::default(),
        });
        element.decorative_sfx = Some(crate::sfx::DecorativeSfxDecision {
            schema_version: crate::sfx::DECORATIVE_SFX_DECISION_SCHEMA_VERSION,
            disposition: crate::sfx::DecorativeSfxDisposition::SkipDifficult,
            review_state: "skipped_difficult_sfx",
            evidence_revision: Revision::new(2),
            decision_revision: Revision::new(3),
            page_id,
            original_ordinal: 1,
            element_id: element.id,
            content_id: element.content_id,
            source_region_id: element.source_region_id.unwrap(),
            source_ocr: element.source.clone().unwrap(),
            source_typography: element.typography.clone().unwrap(),
            source_crop_blake3: "source-crop".to_owned(),
            source_debug_label: "1:SFX".to_owned(),
            classifier: crate::sfx::SfxClassifier {
                kind: "agent_visual_semantic",
                configured_model: None,
                tool_call_id: "call-sfx".to_owned(),
            },
            evidence: crate::sfx::DecorativeSfxEvidence {
                decorative_visual_form: "stylized lettering integrated with impact lines"
                    .to_owned(),
                sound_effect_page_function: "represents an impact sound".to_owned(),
                legibility_and_translation_value:
                    "distorted overlapping glyph strokes prevent a reliable translation".to_owned(),
                exclusion_of_dialogue_caption_ui_and_general_free_text:
                    "outside any speech, caption, or UI container".to_owned(),
            },
            confidence: 0.96,
            rationale: "Positive decorative difficult-SFX classification from source pixels."
                .to_owned(),
            target_translation_owner: None,
            target_render_owner: None,
        });
    }

    fn element(id: EntityId, x: f64, y: f64, text: &str) -> TextElementInspection {
        let geometry = ElementGeometry {
            points: vec![
                ElementPoint { x, y },
                ElementPoint { x: x + 40.0, y },
                ElementPoint {
                    x: x + 40.0,
                    y: y + 20.0,
                },
                ElementPoint { x, y: y + 20.0 },
            ],
            bounds: ElementBounds {
                x,
                y,
                width: 40.0,
                height: 20.0,
            },
        };
        TextElementInspection {
            id,
            content_id: EntityId::new(),
            source_region_id: Some(EntityId::new()),
            source_region_kind: Some("dev.koharu.region.text".to_owned()),
            detected: true,
            required: true,
            text_role: Some("dev.koharu.text.free-text".to_owned()),
            decorative_sfx: None,
            logical_dialogue_memberships: Vec::new(),
            source: Some(SemanticText {
                text: text.to_owned(),
                language: Some("ja".to_owned()),
            }),
            translation: Some(SemanticText {
                text: format!("번역 {text}"),
                language: Some("ko".to_owned()),
            }),
            source_writing_mode: None,
            visibility: ElementVisibility {
                local_visible: true,
                local_opacity: 1.0,
                effective_visible: true,
                effective_opacity: 1.0,
            },
            source_geometry: Some(geometry.clone()),
            text_safe_region: Some(TextSafeRegion {
                id: EntityId::new(),
                kind: "dev.koharu.region.text".to_owned(),
                geometry: geometry.clone(),
                association: None,
            }),
            verified_ui_panel_anchor: None,
            verified_free_dialogue_anchor: None,
            free_dialogue_anchor_assessment: None,
            typography: None,
            layout_kind: TextLayoutKind::Paragraph,
            authored_layout_geometry: Some(geometry.clone()),
            final_scene: FinalSceneElement {
                eligible: true,
                visible: true,
                opacity: 1.0,
                geometry_visible: true,
                glyph_bounds: Some(geometry.bounds),
                layout_bounds: Some(geometry.bounds),
                font_size_px: Some(18.0),
                line_count: Some(1),
                rendered_lines: vec![format!("번역 {text}")],
                diagnostics: Vec::new(),
                glyph_ink: None,
            },
        }
    }

    fn page() -> PageInspection {
        let right = element(EntityId::new(), 70.0, 10.0, "첫째");
        let left = element(EntityId::new(), 10.0, 10.0, "둘째");
        let lower = element(EntityId::new(), 40.0, 60.0, "셋째");
        PageInspection {
            id: EntityId::new(),
            label: "page.png".to_owned(),
            width: 120.0,
            height: 100.0,
            text_elements: vec![lower, left, right],
            detected_panel_candidates: Vec::new(),
            logical_dialogue_groups: Vec::new(),
            render_error: None,
        }
    }

    #[test]
    fn dossier_orders_every_element_and_links_ordinals() {
        let page = page();
        let dossier = build_page_translation_dossier(
            Revision::new(3),
            &page,
            None,
            "ja",
            "ko",
            |_, source| source.clone(),
            RenderedPageReference {
                media_type: "image/webp",
                byte_length: 12,
                blake3: "digest".to_owned(),
            },
        );
        assert_eq!(dossier.elements.len(), 3);
        assert_eq!(
            dossier
                .elements
                .iter()
                .map(|value| value.ordinal)
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert!(
            dossier
                .elements
                .iter()
                .all(|value| !value.short_stable_id.is_empty())
        );
        assert_eq!(dossier.elements[0].adjacency.next_ordinal, Some(2));
        assert_eq!(
            dossier.elements[1].adjacency.previous_element_id,
            Some(dossier.elements[0].element_id)
        );
        assert_eq!(dossier.page_context.ordered_translation_dialogue.len(), 3);
        assert_eq!(dossier.elements[0].layout.rendered_lines, ["번역 첫째"]);
        assert!(
            dossier
                .semantic_assessment
                .contains("no OCR or translation semantic correctness")
        );
    }

    #[test]
    fn debug_overlay_artifact_contains_all_ordinal_labels() {
        let page = page();
        let source = RgbaImage::from_pixel(120, 100, Rgba([255, 255, 255, 255]));
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(source)
            .write_to(&mut encoded, ImageFormat::Png)
            .unwrap();
        let ordinals = page
            .text_elements
            .iter()
            .enumerate()
            .map(|(index, value)| (index + 1, value.id))
            .collect::<Vec<_>>();
        let (overlay, labels) =
            render_page_debug_overlay(encoded.get_ref(), &page, &ordinals).unwrap();
        assert_eq!(labels.len(), 3);
        assert!(labels.iter().all(|label| label.label.contains(':')));
        let directory = tempfile::tempdir().unwrap();
        let artifact = write_page_debug_artifact(
            directory.path(),
            Revision::new(3),
            page.id,
            &overlay,
            labels,
        )
        .unwrap();
        assert_eq!(artifact.media_type, "image/png");
        assert!(Path::new(&artifact.path).is_file());
        assert_eq!(artifact.legend.len(), 6);
        assert_eq!(image::load_from_memory(&overlay).unwrap().width(), 120);
    }

    #[test]
    fn source_dossier_has_authoritative_original_artifact_and_crop_for_every_ordinal() {
        let page = page();
        let source = RgbaImage::from_pixel(120, 100, Rgba([250, 250, 250, 255]));
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(source)
            .write_to(&mut encoded, ImageFormat::Png)
            .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let dossier = build_source_evidence_dossier(
            directory.path(),
            Revision::new(4),
            &page,
            encoded.get_ref(),
            "image/png",
            |_, source| source.clone(),
            |_| Some(0.75),
        )
        .unwrap();

        assert!(
            dossier
                .authority_contract
                .contains("pixels are the authority")
        );
        assert!(dossier.authority_contract.contains("OCR"));
        assert!(Path::new(&dossier.full_page_original.path).is_file());
        assert_eq!(dossier.elements.len(), page.text_elements.len());
        for element in &dossier.elements {
            assert!(Path::new(&element.original_crop.path).is_file());
            assert!(!element.original_crop.blake3.is_empty());
            assert!(element.original_crop.width_px > 0);
            assert!(element.original_crop.height_px > 0);
            assert_eq!(element.ocr_confidence, Some(0.75));
            assert!(
                element
                    .source_debug_label
                    .starts_with(&format!("{}:", element.ordinal))
            );
        }
    }

    #[test]
    fn skipped_difficult_sfx_remains_in_source_and_page_dossiers_but_not_dialogue_context() {
        let mut page = page();
        mark_first_as_skipped_difficult_sfx(&mut page);
        let translated = build_page_translation_dossier(
            Revision::new(5),
            &page,
            None,
            "ja",
            "ko",
            |_, source| source.clone(),
            RenderedPageReference {
                media_type: "image/webp",
                byte_length: 1,
                blake3: "rendered".to_owned(),
            },
        );
        let skipped = translated
            .elements
            .iter()
            .find(|element| element.decorative_sfx.is_some())
            .unwrap();
        assert_eq!(skipped.review_state, "skipped_difficult_sfx");
        assert!(!skipped.required);
        assert!(skipped.current_translation.is_none());
        assert_eq!(translated.page_context.skipped_difficult_sfx_count, 1);
        assert_eq!(translated.page_context.required_content_count, 2);
        assert_eq!(translated.page_context.ordered_source_dialogue.len(), 2);

        let source = RgbaImage::from_pixel(120, 100, Rgba([250, 250, 250, 255]));
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(source)
            .write_to(&mut encoded, ImageFormat::Png)
            .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let source = build_source_evidence_dossier(
            directory.path(),
            Revision::new(5),
            &page,
            encoded.get_ref(),
            "image/png",
            |_, source| source.clone(),
            |_| None,
        )
        .unwrap();
        let skipped = source
            .elements
            .iter()
            .find(|element| element.decorative_sfx.is_some())
            .unwrap();
        assert_eq!(skipped.review_state, "skipped_difficult_sfx");
        assert!(Path::new(&skipped.original_crop.path).is_file());
        assert!(skipped.current_source.is_some());
    }

    #[test]
    fn original_source_overlay_has_unique_ordinal_labels() {
        let mut page = page();
        for (element, id) in page.text_elements.iter_mut().zip([
            "01a05c4b-e355-7202-a1f6-2db1f691d449",
            "01a05c4b-e355-7202-a1f6-2e07c99ca591",
            "01a05c4b-e355-7202-a1f6-2e51e04fd34f",
        ]) {
            element.id = serde_json::from_value(serde_json::Value::String(id.to_owned())).unwrap();
        }
        let source = RgbaImage::from_pixel(120, 100, Rgba([255, 255, 255, 255]));
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(source)
            .write_to(&mut encoded, ImageFormat::Png)
            .unwrap();
        let ordinals = page_reading_order_ordinals(&page);
        let (overlay, labels) =
            render_page_source_debug_overlay(encoded.get_ref(), &page, &ordinals).unwrap();
        let unique_labels = labels
            .iter()
            .map(|label| label.label.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique_labels.len(), page.text_elements.len());
        assert!(
            labels
                .iter()
                .all(|label| label.label.starts_with(&format!("{}:", label.ordinal)))
        );

        let directory = tempfile::tempdir().unwrap();
        let artifact = write_page_source_debug_artifact(
            directory.path(),
            Revision::new(4),
            page.id,
            &overlay,
            labels,
        )
        .unwrap();
        assert_eq!(artifact.legend.len(), 4);
        assert!(artifact.path.contains("source-debug"));
    }

    #[test]
    fn shared_uuid_prefixes_get_unique_labels_correlated_with_dossier_entries() {
        let mut page = page();
        for (element, id) in page.text_elements.iter_mut().zip([
            "01a05c4b-e355-7202-a1f6-2db1f691d449",
            "01a05c4b-e355-7202-a1f6-2e07c99ca591",
            "01a05c4b-e355-7202-a1f6-2e51e04fd34f",
        ]) {
            element.id = serde_json::from_value(serde_json::Value::String(id.to_owned())).unwrap();
        }
        let dossier = build_page_translation_dossier(
            Revision::new(9),
            &page,
            None,
            "ja",
            "ko",
            |_, source| source.clone(),
            RenderedPageReference {
                media_type: "image/webp",
                byte_length: 12,
                blake3: "rendered-digest".to_owned(),
            },
        );
        let ordinals = dossier
            .elements
            .iter()
            .map(|element| (element.ordinal, element.element_id))
            .collect::<Vec<_>>();
        let source = RgbaImage::from_pixel(120, 100, Rgba([255, 255, 255, 255]));
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(source)
            .write_to(&mut encoded, ImageFormat::Png)
            .unwrap();
        let (_, labels) = render_page_debug_overlay(encoded.get_ref(), &page, &ordinals).unwrap();

        let unique_ids = labels
            .iter()
            .map(|label| label.short_stable_id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let unique_labels = labels
            .iter()
            .map(|label| label.label.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique_ids.len(), 3);
        assert_eq!(unique_labels.len(), 3);
        for label in labels {
            let dossier_entry = dossier
                .elements
                .iter()
                .find(|element| element.element_id == label.element_id)
                .unwrap();
            assert_eq!(label.ordinal, dossier_entry.ordinal);
            assert_eq!(label.short_stable_id, dossier_entry.short_stable_id);
            assert_eq!(
                label.label,
                format!(
                    "{}:{}",
                    dossier_entry.ordinal, dossier_entry.short_stable_id
                )
                .to_uppercase()
            );
        }
    }
}
