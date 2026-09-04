use std::collections::VecDeque;

use image::GrayImage;
use koharu_scene::{EntityId, Revision};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::acceptance::{ElementBounds, ElementGeometry, SemanticText};

pub(crate) const UI_TEXT_ROLE: &str = "dev.koharu.text.ui";
pub(crate) const MINIMUM_UI_PANEL_VERIFICATION_CONFIDENCE: f32 = 0.90;
pub(crate) const RASTER_PANEL_DETECTOR: &str = "dev.koharu.agent-e2e.source-raster-ui-panel";
pub(crate) const RASTER_PANEL_DETECTOR_VERSION: &str = "closed-rectangle-v1";
pub(crate) const MINIMUM_PANEL_FONT_PX: f64 = 12.0;
pub(crate) const MINIMUM_PANEL_CLEARANCE_PX: f64 = 4.0;

const DARK_LUMA: u8 = 96;
const LIGHT_LUMA: u8 = 192;
const EDGE_BAND_PX: u32 = 2;
const MIN_SIDE_COVERAGE: f64 = 0.86;
const MIN_CORNER_SUPPORT: f64 = 0.75;
const MIN_INTERIOR_LIGHT_RATIO: f64 = 0.68;
const MIN_BORDER_INTERIOR_CONTRAST: f64 = 72.0;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct DetectedPanelCandidate {
    pub region_id: EntityId,
    pub region_kind: String,
    pub geometry: ElementGeometry,
    pub detection_label: String,
    pub detection_confidence: f32,
    pub detector_producer: String,
    pub detector_model: Option<String>,
    pub raster_evidence: RasterPanelEvidence,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RasterPanelEvidence {
    pub panel_bbox: ElementBounds,
    pub safe_interior_bbox: ElementBounds,
    pub border: BorderMeasurements,
    pub contour: ContourMeasurements,
    pub raster: RasterMeasurements,
    pub source_relationship: PanelSourceRelationship,
    pub detector: DetectorIdentity,
    pub confidence: f32,
    pub rejection_reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BorderMeasurements {
    pub top_dark_coverage: f64,
    pub right_dark_coverage: f64,
    pub bottom_dark_coverage: f64,
    pub left_dark_coverage: f64,
    pub minimum_side_dark_coverage: f64,
    pub mean_border_luma: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ContourMeasurements {
    pub dark_component_pixels: u32,
    pub closed: bool,
    pub corner_support_ratio: f64,
    pub touches_page_edge: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RasterMeasurements {
    pub sampled_interior_pixels: u32,
    pub mean_interior_luma: f64,
    pub interior_light_ratio: f64,
    pub border_interior_contrast: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PanelSourceRelationship {
    pub source_region_id: EntityId,
    pub source_bbox: ElementBounds,
    pub source_inside_safe_interior_ratio: f64,
    pub source_inside_panel_ratio: f64,
    pub relation: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct DetectorIdentity {
    pub producer: &'static str,
    pub version: &'static str,
    pub input: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RasterPanelAssessment {
    pub source_region_id: EntityId,
    pub source_bbox: ElementBounds,
    pub accepted: Option<RasterPanelEvidence>,
    pub rejection_reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct UiPanelAnchorDecision {
    pub schema_version: u32,
    pub decision: &'static str,
    pub evidence_revision: Revision,
    pub decision_revision: Revision,
    pub page_id: EntityId,
    pub original_ordinal: usize,
    pub element_id: EntityId,
    pub content_id: EntityId,
    pub source_region_id: EntityId,
    pub source_ocr: SemanticText,
    pub source_crop_blake3: String,
    pub source_debug_label: String,
    pub panel: DetectedPanelCandidate,
    pub source_containment_ratio: f64,
    pub source_intersection_ratio: f64,
    pub classifier: UiPanelClassifier,
    pub evidence: UiPanelEvidence,
    pub confidence: f32,
    pub association_reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct UiPanelClassifier {
    pub kind: &'static str,
    pub configured_model: Option<String>,
    pub tool_call_id: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UiPanelEvidence {
    pub ui_role_and_function: String,
    pub finite_visible_panel_or_screen: String,
    pub source_to_panel_relation: String,
    pub text_safe_interior: String,
    pub visual_or_detection_provenance: String,
}

#[derive(Clone, Copy)]
struct ComponentBounds {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    pixels: u32,
}

pub(crate) fn assess_source_raster_panel(
    image: &GrayImage,
    source_region_id: EntityId,
    source_bbox: ElementBounds,
) -> RasterPanelAssessment {
    let mut assessment = RasterPanelAssessment {
        source_region_id,
        source_bbox,
        accepted: None,
        rejection_reasons: Vec::new(),
    };
    if !valid_bounds(source_bbox)
        || source_bbox.x < 0.0
        || source_bbox.y < 0.0
        || source_bbox.x + source_bbox.width > f64::from(image.width())
        || source_bbox.y + source_bbox.height > f64::from(image.height())
    {
        assessment
            .rejection_reasons
            .push("non_finite_or_out_of_raster_source_bbox".to_owned());
        return assessment;
    }

    let mut enclosing = dark_components(image)
        .into_iter()
        .filter(|component| component_encloses_source(*component, source_bbox))
        .filter(|component| plausible_panel_scale(*component, source_bbox))
        .collect::<Vec<_>>();
    enclosing.sort_by_key(|component| {
        (component.max_x - component.min_x + 1) * (component.max_y - component.min_y + 1)
    });
    if enclosing.is_empty() {
        assessment
            .rejection_reasons
            .push("no_enclosing_dark_contour".to_owned());
        return assessment;
    }

    let mut best_rejection = Vec::new();
    for component in enclosing {
        let (evidence, reasons) =
            measure_component(image, component, source_region_id, source_bbox);
        if reasons.is_empty() {
            assessment.accepted = Some(evidence);
            return assessment;
        }
        if best_rejection.is_empty() || reasons.len() < best_rejection.len() {
            best_rejection = reasons;
        }
    }
    assessment.rejection_reasons = best_rejection;
    assessment
}

pub(crate) fn raster_candidate_supports_ui_role(
    evidence: &RasterPanelEvidence,
    source_region_id: EntityId,
    source_bbox: ElementBounds,
    ui_role_and_function: &str,
) -> Result<(), &'static str> {
    if ui_role_and_function.trim().is_empty() {
        return Err("missing_ui_role_evidence");
    }
    validate_raster_panel_evidence(evidence, source_region_id, source_bbox)
}

pub(crate) fn validate_raster_panel_evidence(
    evidence: &RasterPanelEvidence,
    source_region_id: EntityId,
    source_bbox: ElementBounds,
) -> Result<(), &'static str> {
    if !valid_bounds(source_bbox)
        || !valid_bounds(evidence.panel_bbox)
        || !valid_bounds(evidence.safe_interior_bbox)
        || containment_ratio(evidence.safe_interior_bbox, evidence.panel_bbox) < 0.99
    {
        return Err("non_finite_or_invalid_panel_safe_interior_geometry");
    }
    if evidence.source_relationship.source_region_id != source_region_id
        || evidence.source_relationship.source_bbox != source_bbox
    {
        return Err("raster_candidate_source_identity_mismatch");
    }
    if evidence.detector.producer != RASTER_PANEL_DETECTOR
        || evidence.detector.version != RASTER_PANEL_DETECTOR_VERSION
        || evidence.detector.input != "original_source_pixels"
    {
        return Err("unrecognized_raster_detector_provenance");
    }
    if !evidence.contour.closed
        || evidence.contour.touches_page_edge
        || evidence.border.minimum_side_dark_coverage < MIN_SIDE_COVERAGE
    {
        return Err("border_not_closed");
    }
    if evidence.raster.interior_light_ratio < MIN_INTERIOR_LIGHT_RATIO
        || evidence.raster.border_interior_contrast < MIN_BORDER_INTERIOR_CONTRAST
    {
        return Err("interior_not_materially_distinct");
    }
    if evidence
        .source_relationship
        .source_inside_safe_interior_ratio
        < 0.99
    {
        return Err("source_not_contained_in_safe_interior");
    }
    let minimum_room = MINIMUM_PANEL_FONT_PX + 2.0 * MINIMUM_PANEL_CLEARANCE_PX;
    if evidence.safe_interior_bbox.width < minimum_room
        || evidence.safe_interior_bbox.height < minimum_room
    {
        return Err("insufficient_text_room");
    }
    if !evidence.rejection_reasons.is_empty()
        || !evidence.confidence.is_finite()
        || evidence.confidence < MINIMUM_UI_PANEL_VERIFICATION_CONFIDENCE
    {
        return Err("raster_panel_confidence_below_threshold");
    }
    Ok(())
}

fn measure_component(
    image: &GrayImage,
    component: ComponentBounds,
    source_region_id: EntityId,
    source_bbox: ElementBounds,
) -> (RasterPanelEvidence, Vec<String>) {
    let panel_bbox = ElementBounds {
        x: f64::from(component.min_x),
        y: f64::from(component.min_y),
        width: f64::from(component.max_x - component.min_x + 1),
        height: f64::from(component.max_y - component.min_y + 1),
    };
    let inset = f64::from(
        EDGE_BAND_PX.min(
            ((component.max_x - component.min_x + 1).min(component.max_y - component.min_y + 1)
                / 4)
            .max(1),
        ),
    );
    let safe_interior_bbox = ElementBounds {
        x: panel_bbox.x + inset,
        y: panel_bbox.y + inset,
        width: (panel_bbox.width - inset * 2.0).max(0.0),
        height: (panel_bbox.height - inset * 2.0).max(0.0),
    };
    let top = horizontal_coverage(image, component.min_y, component.min_x, component.max_x);
    let bottom = horizontal_coverage(image, component.max_y, component.min_x, component.max_x);
    let left = vertical_coverage(image, component.min_x, component.min_y, component.max_y);
    let right = vertical_coverage(image, component.max_x, component.min_y, component.max_y);
    let minimum_side = top.min(right).min(bottom).min(left);
    let corner_support_ratio = corner_support(image, component);
    let touches_page_edge = component.min_x == 0
        || component.min_y == 0
        || component.max_x + 1 == image.width()
        || component.max_y + 1 == image.height();
    let (mean_border_luma, border_samples) = mean_border_luma(image, component);
    let (mean_interior_luma, interior_light_ratio, interior_samples) =
        interior_measurements(image, safe_interior_bbox, source_bbox);
    let border_interior_contrast = mean_interior_luma - mean_border_luma;
    let source_inside_safe_interior_ratio = containment_ratio(source_bbox, safe_interior_bbox);
    let source_inside_panel_ratio = containment_ratio(source_bbox, panel_bbox);
    let closed = minimum_side >= MIN_SIDE_COVERAGE
        && corner_support_ratio >= MIN_CORNER_SUPPORT
        && !touches_page_edge;
    let contrast_score = (border_interior_contrast / 128.0).clamp(0.0, 1.0);
    let confidence = (0.55 * minimum_side
        + 0.20 * corner_support_ratio
        + 0.15 * contrast_score
        + 0.10 * interior_light_ratio) as f32;
    let mut rejection_reasons = Vec::new();
    if !closed {
        rejection_reasons.push("border_not_closed".to_owned());
    }
    if interior_samples == 0
        || border_samples == 0
        || interior_light_ratio < MIN_INTERIOR_LIGHT_RATIO
        || border_interior_contrast < MIN_BORDER_INTERIOR_CONTRAST
    {
        rejection_reasons.push("interior_not_materially_distinct".to_owned());
    }
    if source_inside_safe_interior_ratio < 0.99 {
        rejection_reasons.push("source_not_contained_in_safe_interior".to_owned());
    }
    let minimum_room = MINIMUM_PANEL_FONT_PX + 2.0 * MINIMUM_PANEL_CLEARANCE_PX;
    if safe_interior_bbox.width < minimum_room || safe_interior_bbox.height < minimum_room {
        rejection_reasons.push("insufficient_text_room".to_owned());
    }
    if confidence < MINIMUM_UI_PANEL_VERIFICATION_CONFIDENCE {
        rejection_reasons.push("raster_panel_confidence_below_threshold".to_owned());
    }
    let evidence = RasterPanelEvidence {
        panel_bbox,
        safe_interior_bbox,
        border: BorderMeasurements {
            top_dark_coverage: top,
            right_dark_coverage: right,
            bottom_dark_coverage: bottom,
            left_dark_coverage: left,
            minimum_side_dark_coverage: minimum_side,
            mean_border_luma,
        },
        contour: ContourMeasurements {
            dark_component_pixels: component.pixels,
            closed,
            corner_support_ratio,
            touches_page_edge,
        },
        raster: RasterMeasurements {
            sampled_interior_pixels: interior_samples,
            mean_interior_luma,
            interior_light_ratio,
            border_interior_contrast,
        },
        source_relationship: PanelSourceRelationship {
            source_region_id,
            source_bbox,
            source_inside_safe_interior_ratio,
            source_inside_panel_ratio,
            relation: "exact_source_bbox_contained_in_detected_safe_interior",
        },
        detector: DetectorIdentity {
            producer: RASTER_PANEL_DETECTOR,
            version: RASTER_PANEL_DETECTOR_VERSION,
            input: "original_source_pixels",
        },
        confidence,
        rejection_reasons: rejection_reasons.clone(),
    };
    (evidence, rejection_reasons)
}

fn dark_components(image: &GrayImage) -> Vec<ComponentBounds> {
    let width = image.width();
    let height = image.height();
    let mut visited = vec![false; width as usize * height as usize];
    let mut components = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let index = y as usize * width as usize + x as usize;
            if visited[index] || image.get_pixel(x, y)[0] > DARK_LUMA {
                continue;
            }
            visited[index] = true;
            let mut queue = VecDeque::from([(x, y)]);
            let mut bounds = ComponentBounds {
                min_x: x,
                min_y: y,
                max_x: x,
                max_y: y,
                pixels: 0,
            };
            while let Some((current_x, current_y)) = queue.pop_front() {
                bounds.min_x = bounds.min_x.min(current_x);
                bounds.min_y = bounds.min_y.min(current_y);
                bounds.max_x = bounds.max_x.max(current_x);
                bounds.max_y = bounds.max_y.max(current_y);
                bounds.pixels += 1;
                for next_y in current_y.saturating_sub(1)..=(current_y + 1).min(height - 1) {
                    for next_x in current_x.saturating_sub(1)..=(current_x + 1).min(width - 1) {
                        let next = next_y as usize * width as usize + next_x as usize;
                        if !visited[next] && image.get_pixel(next_x, next_y)[0] <= DARK_LUMA {
                            visited[next] = true;
                            queue.push_back((next_x, next_y));
                        }
                    }
                }
            }
            if bounds.pixels >= 8 {
                components.push(bounds);
            }
        }
    }
    components
}

fn component_encloses_source(component: ComponentBounds, source: ElementBounds) -> bool {
    f64::from(component.min_x) + 1.0 <= source.x
        && f64::from(component.min_y) + 1.0 <= source.y
        && f64::from(component.max_x) >= source.x + source.width
        && f64::from(component.max_y) >= source.y + source.height
}

fn plausible_panel_scale(component: ComponentBounds, source: ElementBounds) -> bool {
    let width = f64::from(component.max_x - component.min_x + 1);
    let height = f64::from(component.max_y - component.min_y + 1);
    let maximum_width = (source.width * 5.0).max(192.0);
    let maximum_height = (source.height * 6.0).max(128.0);
    width >= 8.0 && height >= 8.0 && width <= maximum_width && height <= maximum_height
}

fn horizontal_coverage(image: &GrayImage, y: u32, min_x: u32, max_x: u32) -> f64 {
    let supported = (min_x..=max_x)
        .filter(|x| {
            (y.saturating_sub(EDGE_BAND_PX)..=(y + EDGE_BAND_PX).min(image.height() - 1))
                .any(|sample_y| image.get_pixel(*x, sample_y)[0] <= DARK_LUMA)
        })
        .count();
    supported as f64 / f64::from(max_x - min_x + 1)
}

fn vertical_coverage(image: &GrayImage, x: u32, min_y: u32, max_y: u32) -> f64 {
    let supported = (min_y..=max_y)
        .filter(|y| {
            (x.saturating_sub(EDGE_BAND_PX)..=(x + EDGE_BAND_PX).min(image.width() - 1))
                .any(|sample_x| image.get_pixel(sample_x, *y)[0] <= DARK_LUMA)
        })
        .count();
    supported as f64 / f64::from(max_y - min_y + 1)
}

fn corner_support(image: &GrayImage, component: ComponentBounds) -> f64 {
    let radius = EDGE_BAND_PX + 1;
    let corners = [
        (component.min_x, component.min_y),
        (component.max_x, component.min_y),
        (component.max_x, component.max_y),
        (component.min_x, component.max_y),
    ];
    corners
        .into_iter()
        .filter(|(x, y)| {
            (y.saturating_sub(radius)..=(*y + radius).min(image.height() - 1)).any(|sample_y| {
                (x.saturating_sub(radius)..=(*x + radius).min(image.width() - 1))
                    .any(|sample_x| image.get_pixel(sample_x, sample_y)[0] <= DARK_LUMA)
            })
        })
        .count() as f64
        / 4.0
}

fn mean_border_luma(image: &GrayImage, component: ComponentBounds) -> (f64, u32) {
    let mut sum = 0_u64;
    let mut count = 0_u32;
    for y in component.min_y..=component.max_y {
        for x in component.min_x..=component.max_x {
            let near_edge = x - component.min_x < EDGE_BAND_PX
                || component.max_x - x < EDGE_BAND_PX
                || y - component.min_y < EDGE_BAND_PX
                || component.max_y - y < EDGE_BAND_PX;
            if near_edge {
                sum += u64::from(image.get_pixel(x, y)[0]);
                count += 1;
            }
        }
    }
    (sum as f64 / f64::from(count.max(1)), count)
}

fn interior_measurements(
    image: &GrayImage,
    interior: ElementBounds,
    source: ElementBounds,
) -> (f64, f64, u32) {
    let min_x = interior.x.ceil().max(0.0) as u32;
    let min_y = interior.y.ceil().max(0.0) as u32;
    let max_x = (interior.x + interior.width)
        .floor()
        .min(f64::from(image.width())) as u32;
    let max_y = (interior.y + interior.height)
        .floor()
        .min(f64::from(image.height())) as u32;
    let mut sum = 0_u64;
    let mut light = 0_u32;
    let mut count = 0_u32;
    for y in min_y..max_y {
        for x in min_x..max_x {
            if f64::from(x + 1) > source.x - 1.0
                && f64::from(x) < source.x + source.width + 1.0
                && f64::from(y + 1) > source.y - 1.0
                && f64::from(y) < source.y + source.height + 1.0
            {
                continue;
            }
            let value = image.get_pixel(x, y)[0];
            sum += u64::from(value);
            light += u32::from(value >= LIGHT_LUMA);
            count += 1;
        }
    }
    if count == 0 {
        return (0.0, 0.0, 0);
    }
    (
        sum as f64 / f64::from(count),
        f64::from(light) / f64::from(count),
        count,
    )
}

fn containment_ratio(source: ElementBounds, target: ElementBounds) -> f64 {
    let left = source.x.max(target.x);
    let top = source.y.max(target.y);
    let right = (source.x + source.width).min(target.x + target.width);
    let bottom = (source.y + source.height).min(target.y + target.height);
    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    intersection / (source.width * source.height)
}

fn valid_bounds(bounds: ElementBounds) -> bool {
    [bounds.x, bounds.y, bounds.width, bounds.height]
        .into_iter()
        .all(f64::is_finite)
        && bounds.width > 0.0
        && bounds.height > 0.0
}

#[cfg(test)]
pub(crate) fn test_raster_panel_evidence(
    source_region_id: EntityId,
    source_bbox: ElementBounds,
    panel_bbox: ElementBounds,
) -> RasterPanelEvidence {
    RasterPanelEvidence {
        panel_bbox,
        safe_interior_bbox: ElementBounds {
            x: panel_bbox.x + 2.0,
            y: panel_bbox.y + 2.0,
            width: panel_bbox.width - 4.0,
            height: panel_bbox.height - 4.0,
        },
        border: BorderMeasurements {
            top_dark_coverage: 1.0,
            right_dark_coverage: 1.0,
            bottom_dark_coverage: 1.0,
            left_dark_coverage: 1.0,
            minimum_side_dark_coverage: 1.0,
            mean_border_luma: 20.0,
        },
        contour: ContourMeasurements {
            dark_component_pixels: 400,
            closed: true,
            corner_support_ratio: 1.0,
            touches_page_edge: false,
        },
        raster: RasterMeasurements {
            sampled_interior_pixels: 2_000,
            mean_interior_luma: 240.0,
            interior_light_ratio: 0.98,
            border_interior_contrast: 220.0,
        },
        source_relationship: PanelSourceRelationship {
            source_region_id,
            source_bbox,
            source_inside_safe_interior_ratio: 1.0,
            source_inside_panel_ratio: 1.0,
            relation: "exact_source_bbox_contained_in_detected_safe_interior",
        },
        detector: DetectorIdentity {
            producer: RASTER_PANEL_DETECTOR,
            version: RASTER_PANEL_DETECTOR_VERSION,
            input: "original_source_pixels",
        },
        confidence: 0.98,
        rejection_reasons: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use image::{GrayImage, Luma};

    use super::*;

    fn outlined_fixture(width: u32, height: u32, open_right: bool) -> GrayImage {
        let mut image = GrayImage::from_pixel(160, 120, Luma([235]));
        for x in 20..20 + width {
            image.put_pixel(x, 20, Luma([12]));
            image.put_pixel(x, 20 + height - 1, Luma([12]));
        }
        for y in 20..20 + height {
            image.put_pixel(20, y, Luma([12]));
            if !open_right {
                image.put_pixel(20 + width - 1, y, Luma([12]));
            }
        }
        for y in 45..55 {
            for x in (65..95).step_by(6) {
                image.put_pixel(x, y, Luma([30]));
            }
        }
        image
    }

    fn source_bounds() -> ElementBounds {
        ElementBounds {
            x: 64.0,
            y: 44.0,
            width: 32.0,
            height: 12.0,
        }
    }

    #[test]
    fn closed_source_raster_display_produces_verified_ui_candidate() {
        let source = EntityId::new();
        let assessment =
            assess_source_raster_panel(&outlined_fixture(120, 70, false), source, source_bounds());
        let evidence = assessment
            .accepted
            .expect("closed display must be detected");
        assert!(evidence.panel_bbox.x >= 19.0 && evidence.panel_bbox.x <= 20.0);
        assert!(evidence.panel_bbox.y >= 19.0 && evidence.panel_bbox.y <= 20.0);
        assert!(evidence.contour.closed);
        assert!(evidence.raster.border_interior_contrast >= MIN_BORDER_INTERIOR_CONTRAST);
        assert!(
            raster_candidate_supports_ui_role(
                &evidence,
                source,
                source_bounds(),
                "device fault status label",
            )
            .is_ok()
        );
    }

    #[test]
    fn open_or_edgeless_shape_is_rejected() {
        let source = EntityId::new();
        let open =
            assess_source_raster_panel(&outlined_fixture(120, 70, true), source, source_bounds());
        assert!(open.accepted.is_none());
        assert!(open.rejection_reasons.iter().any(|reason| {
            reason == "border_not_closed" || reason == "no_enclosing_dark_contour"
        }));

        let mut edgeless = GrayImage::from_pixel(160, 120, Luma([235]));
        for y in 45..55 {
            edgeless.put_pixel(70, y, Luma([30]));
        }
        let edgeless = assess_source_raster_panel(&edgeless, source, source_bounds());
        assert!(edgeless.accepted.is_none());
        assert_eq!(edgeless.rejection_reasons, ["no_enclosing_dark_contour"]);
    }

    #[test]
    fn raster_geometry_without_ui_role_is_rejected() {
        let source = EntityId::new();
        let evidence =
            assess_source_raster_panel(&outlined_fixture(120, 70, false), source, source_bounds())
                .accepted
                .unwrap();
        assert_eq!(
            raster_candidate_supports_ui_role(&evidence, source, source_bounds(), ""),
            Err("missing_ui_role_evidence")
        );
    }

    #[test]
    fn panel_without_minimum_font_and_clearance_room_is_rejected() {
        let source = EntityId::new();
        let source_bbox = ElementBounds {
            x: 25.0,
            y: 23.0,
            width: 8.0,
            height: 8.0,
        };
        let assessment =
            assess_source_raster_panel(&outlined_fixture(18, 18, false), source, source_bbox);
        assert!(assessment.accepted.is_none());
        assert!(
            assessment
                .rejection_reasons
                .contains(&"insufficient_text_room".to_owned())
        );
    }
}
