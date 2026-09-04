use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use koharu_scene::{
    BubbleRegion, EntityId, EntityOrigin, Generation, Inside, LogicalDialogue,
    LogicalDialogueMember, Origin, ProducerId, RecognizedFrom, RegionSpec, RemovePolicy,
    SourceText, TextDirection, TextLayoutKind, TextRegion, Translation, Typography, Visibility,
    WritingMode,
};
use koharu_translator::Language;
use serde::Serialize;

use super::{detection, finish};
use crate::scope::geometry_extents;

const MODEL: &str = "source-region-preprocess-v1";
const MERGE_OVER_SMALLER_MINIMUM: f64 = 0.90;
const TOKEN_CONTAINMENT_MINIMUM: f64 = 0.80;
const MINIMUM_FUZZY_TOKENS: usize = 4;
const MAXIMUM_AUTO_FIT_FONT_SIZE: f32 = 300.0;
const RELIABLE_TARGET_REGION_CONTAINMENT: f64 = 0.90;
const DIALOGUE_ROLE: &str = "dev.koharu.text.dialogue";
const SKIPPED_DIFFICULT_SFX_ROLE: &str = "dev.koharu.text.skipped-difficult-sfx";
const VERTICAL_COLUMN_OVERLAP_MINIMUM: f64 = 0.50;

#[derive(Clone, Debug, Serialize)]
pub struct PreprocessingReport {
    pub schema_version: u32,
    pub page_id: EntityId,
    pub merges: Vec<SourceRegionMerge>,
    pub logical_dialogue_groups: Vec<LogicalDialogueGroup>,
    pub style_adjustments: Vec<TypesettingAdjustment>,
}

impl PreprocessingReport {
    pub fn adjusted(&self) -> bool {
        !self.merges.is_empty()
            || !self.logical_dialogue_groups.is_empty()
            || !self.style_adjustments.is_empty()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LogicalDialogueGroup {
    pub group_id: EntityId,
    pub primary_render_element_id: EntityId,
    pub target_region_id: EntityId,
    pub members: Vec<LogicalDialogueGroupMember>,
    pub logical_source_text: String,
    pub reading_order: &'static str,
    pub reason: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct LogicalDialogueGroupMember {
    pub ordinal: u32,
    pub element_id: EntityId,
    pub content_id: EntityId,
    pub source_region_id: EntityId,
    pub source_text: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceRegionMerge {
    pub kept_element_id: EntityId,
    pub kept_content_id: EntityId,
    pub kept_source_region_id: EntityId,
    pub removed_element_id: EntityId,
    pub removed_content_id: EntityId,
    pub removed_source_region_id: EntityId,
    pub kept_source_text: String,
    pub removed_source_text: String,
    pub source_region_overlap_over_smaller: f64,
    pub source_token_sequence_containment: f64,
    pub reason: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct TypesettingAdjustment {
    pub element_id: EntityId,
    pub source_region_id: EntityId,
    pub previous_max_font_size_px: Option<f32>,
    pub configured_max_font_size_px: f32,
    pub previous_writing_mode: Option<WritingMode>,
    pub configured_writing_mode: Option<WritingMode>,
    pub reason: &'static str,
}

#[derive(Clone)]
struct Candidate {
    layer: EntityId,
    content: EntityId,
    region: EntityId,
    bounds: [f64; 4],
    area: f64,
    source: SourceText,
    tokens: Vec<String>,
    typography: Typography,
    target_region: Option<EntityId>,
    target_bounds: Option<[f64; 4]>,
    verified_vertical_japanese_dialogue: bool,
}

pub fn prepare_translation_inputs(
    snapshot: &koharu_scene::Snapshot,
    page: EntityId,
    target_language: Language,
) -> Result<(koharu_scene::Patch, PreprocessingReport)> {
    let candidates = candidates(snapshot, page)?;
    let logical_dialogue_groups = logical_dialogue_groups(&candidates, target_language);
    let grouped = logical_dialogue_groups
        .iter()
        .flat_map(|group| group.members.iter().map(|member| member.element_id))
        .collect::<BTreeSet<_>>();
    let group_primaries = logical_dialogue_groups
        .iter()
        .map(|group| group.primary_render_element_id)
        .collect::<BTreeSet<_>>();
    let merge_candidates = candidates
        .iter()
        .filter(|candidate| !grouped.contains(&candidate.layer))
        .cloned()
        .collect::<Vec<_>>();
    let merges = justified_merges(&merge_candidates);
    let removed = merges
        .iter()
        .map(|merge| merge.removed_element_id)
        .collect::<BTreeSet<_>>();
    let generation = Generation {
        producer: ProducerId::new(detection::PRODUCER)?,
        model: Some(MODEL.to_owned()),
        confidence: None,
    };
    let mut edit = snapshot.edit_as(generation.clone());
    edit.observe_subtree(page)?;

    for group in &logical_dialogue_groups {
        edit.set(
            group.group_id,
            &LogicalDialogue {
                origin: Origin::Generated(generation.clone()),
                target_region_id: group.target_region_id,
                members: group
                    .members
                    .iter()
                    .map(|member| LogicalDialogueMember {
                        ordinal: member.ordinal,
                        element_id: member.element_id,
                        content_id: member.content_id,
                        source_region_id: member.source_region_id,
                    })
                    .collect(),
            },
        )?;
        for member in group.members.iter().skip(1) {
            edit.set(
                member.element_id,
                &Visibility {
                    origin: Origin::Generated(generation.clone()),
                    visible: false,
                    opacity: 1.0,
                },
            )?;
        }
    }

    for merge in &merges {
        edit.remove_entity(merge.removed_element_id, RemovePolicy::Cascade)?;
        edit.remove_entity(merge.removed_content_id, RemovePolicy::Cascade)?;
        edit.remove_entity(merge.removed_source_region_id, RemovePolicy::Cascade)?;
    }

    let mut style_adjustments = Vec::new();
    for candidate in candidates {
        if removed.contains(&candidate.layer)
            || (grouped.contains(&candidate.layer) && !group_primaries.contains(&candidate.layer))
            || candidate.typography.size.is_some()
        {
            continue;
        }
        let Some(layer) = snapshot.text_layer(candidate.layer).ok() else {
            continue;
        };
        if layer.layout()?.kind != TextLayoutKind::Paragraph || !candidate.typography.auto_fit {
            continue;
        }
        let previous_writing_mode = candidate.typography.writing_mode;
        let configured_writing_mode = if target_language == Language::Korean {
            Some(WritingMode::Horizontal)
        } else {
            previous_writing_mode
        };
        let [left, top, right, bottom] = logical_dialogue_groups
            .iter()
            .find(|group| group.primary_render_element_id == candidate.layer)
            .and_then(|_| candidate.target_bounds)
            .unwrap_or(candidate.bounds);
        let inline_extent = match configured_writing_mode {
            Some(WritingMode::Vertical) => bottom - top,
            Some(WritingMode::Horizontal) | None => right - left,
        };
        if !inline_extent.is_finite() || inline_extent <= 0.0 {
            continue;
        }
        let maximum = (inline_extent as f32).min(MAXIMUM_AUTO_FIT_FONT_SIZE);
        let mut typography = candidate.typography;
        typography.size = Some(maximum);
        typography.writing_mode = configured_writing_mode;
        edit.set(candidate.layer, &typography)?;
        style_adjustments.push(TypesettingAdjustment {
            element_id: candidate.layer,
            source_region_id: candidate.region,
            previous_max_font_size_px: None,
            configured_max_font_size_px: maximum,
            previous_writing_mode,
            configured_writing_mode,
            reason:
                "generated paragraph uses its source-region inline extent as the auto-fit maximum",
        });
    }

    Ok((
        finish(edit)?,
        PreprocessingReport {
            schema_version: 2,
            page_id: page,
            merges,
            logical_dialogue_groups,
            style_adjustments,
        },
    ))
}

fn candidates(snapshot: &koharu_scene::Snapshot, page: EntityId) -> Result<Vec<Candidate>> {
    let Some(group) = snapshot.page(page)?.text_group()? else {
        return Ok(Vec::new());
    };
    let mut candidates = Vec::new();
    for layer in group.text_layers()? {
        let layer_id = layer.id();
        if !owned_by_detection(snapshot.component::<EntityOrigin>(layer_id)?.as_ref()) {
            continue;
        }
        let content = layer.content()?;
        if content
            .role()?
            .is_some_and(|role| role.role == SKIPPED_DIFFICULT_SFX_ROLE)
        {
            continue;
        }
        if !owned_by_detection(snapshot.component::<EntityOrigin>(content.id())?.as_ref()) {
            continue;
        }
        let Some(region) = content.source_region()? else {
            continue;
        };
        if region.region()?.kind != TextRegion::kind()
            || region.detection()?.is_none()
            || !owned_by_detection(snapshot.component::<EntityOrigin>(region.id())?.as_ref())
            || snapshot
                .relations_to_as::<koharu_scene::Presents>(content.id())
                .count()
                != 1
            || snapshot
                .relations_to_as::<RecognizedFrom>(region.id())
                .count()
                != 1
        {
            continue;
        }
        let Some(source) = content.source()? else {
            continue;
        };
        let Some(typography) = layer.typography()? else {
            continue;
        };
        if !matches!(typography.origin, Origin::Generated(ref owner) if owner.producer.as_str() == detection::PRODUCER)
        {
            continue;
        }
        let geometry = region.geometry()?;
        let Some((left, top, right, bottom)) = geometry_extents(&geometry) else {
            continue;
        };
        let area = (right - left).max(0.0) * (bottom - top).max(0.0);
        if area <= 0.0 {
            continue;
        }
        let target = layer.balloon_target()?;
        let target_region = target.map(|target| target.id());
        let target_bounds = target
            .map(|target| {
                let region = target.region()?;
                let geometry = target.geometry()?;
                Ok::<_, anyhow::Error>((region, geometry_extents(&geometry)))
            })
            .transpose()?
            .and_then(|(region, bounds)| (region.kind == BubbleRegion::kind()).then_some(bounds))
            .flatten()
            .filter(|(left, top, right, bottom)| right > left && bottom > top)
            .map(|(left, top, right, bottom)| [left, top, right, bottom]);
        let source_inside_target = target_region.is_some_and(|target| {
            snapshot
                .relations_from_as::<Inside>(region.id())
                .any(|relation| relation.value().target == target)
        });
        let association_confidence = target_bounds.map(|target| {
            let intersection = (right.min(target[2]) - left.max(target[0])).max(0.0)
                * (bottom.min(target[3]) - top.max(target[1])).max(0.0);
            intersection / area
        });
        let vertical_ocr = region
            .ocr()?
            .is_some_and(|analysis| analysis.direction == TextDirection::Vertical);
        let japanese_source = source
            .language
            .as_ref()
            .is_some_and(|language| is_japanese(language.as_str()));
        let dialogue_role = content
            .role()?
            .is_some_and(|role| role.role == DIALOGUE_ROLE);
        let no_translation = snapshot.component::<Translation>(content.id())?.is_none();
        let generated_or_default_visibility = layer
            .visibility()?
            .is_none_or(|visibility| matches!(visibility.origin, Origin::Generated(_)));
        candidates.push(Candidate {
            layer: layer_id,
            content: content.id(),
            region: region.id(),
            bounds: [left, top, right, bottom],
            area,
            tokens: source_tokens(&source.text.value),
            source,
            typography,
            target_region,
            target_bounds,
            verified_vertical_japanese_dialogue: target_region.is_some()
                && target_bounds.is_some()
                && source_inside_target
                && association_confidence
                    .is_some_and(|value| value >= RELIABLE_TARGET_REGION_CONTAINMENT)
                && vertical_ocr
                && japanese_source
                && dialogue_role
                && no_translation
                && generated_or_default_visibility,
        });
    }
    Ok(candidates)
}

fn logical_dialogue_groups(
    candidates: &[Candidate],
    target_language: Language,
) -> Vec<LogicalDialogueGroup> {
    if target_language != Language::Korean {
        return Vec::new();
    }
    let mut by_target = BTreeMap::<EntityId, Vec<&Candidate>>::new();
    for candidate in candidates
        .iter()
        .filter(|candidate| candidate.verified_vertical_japanese_dialogue)
    {
        by_target
            .entry(candidate.target_region.expect("verified target exists"))
            .or_default()
            .push(candidate);
    }
    by_target
        .into_iter()
        .filter_map(|(target_region_id, mut members)| {
            if members.len() < 2 {
                return None;
            }
            members = japanese_vertical_reading_order(members);
            let primary = members[0];
            let members = members
                .into_iter()
                .enumerate()
                .map(|(index, member)| LogicalDialogueGroupMember {
                    ordinal: index as u32 + 1,
                    element_id: member.layer,
                    content_id: member.content,
                    source_region_id: member.region,
                    source_text: member.source.text.value.clone(),
                })
                .collect::<Vec<_>>();
            Some(LogicalDialogueGroup {
                group_id: primary.content,
                primary_render_element_id: primary.layer,
                target_region_id,
                logical_source_text: members
                    .iter()
                    .map(|member| member.source_text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
                members,
                reading_order: "japanese_vertical_right_to_left_then_top_to_bottom_with_stable_id_tiebreak",
                reason: "multiple vertical Japanese dialogue regions share the same explicit finite bubble through flows-in and inside relations with reliable containment",
            })
        })
        .collect()
}

fn japanese_vertical_reading_order(mut members: Vec<&Candidate>) -> Vec<&Candidate> {
    members.sort_by(|first, second| {
        second.bounds[2]
            .total_cmp(&first.bounds[2])
            .then_with(|| second.bounds[0].total_cmp(&first.bounds[0]))
            .then_with(|| first.bounds[1].total_cmp(&second.bounds[1]))
            .then_with(|| first.layer.cmp(&second.layer))
    });
    let mut columns = Vec::<([f64; 2], Vec<&Candidate>)>::new();
    for member in members {
        let width = member.bounds[2] - member.bounds[0];
        let matching = columns
            .iter()
            .enumerate()
            .filter_map(|(index, (extent, _))| {
                let overlap =
                    (member.bounds[2].min(extent[1]) - member.bounds[0].max(extent[0])).max(0.0);
                let ratio = overlap / width.min(extent[1] - extent[0]);
                (ratio >= VERTICAL_COLUMN_OVERLAP_MINIMUM).then_some((index, ratio))
            })
            .max_by(|(first_index, first), (second_index, second)| {
                first
                    .total_cmp(second)
                    .then_with(|| second_index.cmp(first_index))
            })
            .map(|(index, _)| index);
        if let Some(index) = matching {
            columns[index].0[0] = columns[index].0[0].min(member.bounds[0]);
            columns[index].0[1] = columns[index].0[1].max(member.bounds[2]);
            columns[index].1.push(member);
        } else {
            columns.push(([member.bounds[0], member.bounds[2]], vec![member]));
        }
    }
    columns.sort_by(|(first, first_members), (second, second_members)| {
        second[1]
            .total_cmp(&first[1])
            .then_with(|| second[0].total_cmp(&first[0]))
            .then_with(|| first_members[0].layer.cmp(&second_members[0].layer))
    });
    columns
        .into_iter()
        .flat_map(|(_, mut column)| {
            column.sort_by(|first, second| {
                first.bounds[1]
                    .total_cmp(&second.bounds[1])
                    .then_with(|| second.bounds[0].total_cmp(&first.bounds[0]))
                    .then_with(|| first.layer.cmp(&second.layer))
            });
            column
        })
        .collect()
}

fn is_japanese(language: &str) -> bool {
    language
        .split(['-', '_'])
        .next()
        .is_some_and(|primary| primary.eq_ignore_ascii_case("ja"))
}

fn owned_by_detection(origin: Option<&EntityOrigin>) -> bool {
    matches!(origin, Some(EntityOrigin { origin: Origin::Generated(owner) }) if owner.producer.as_str() == detection::PRODUCER)
}

fn justified_merges(candidates: &[Candidate]) -> Vec<SourceRegionMerge> {
    let mut order = (0..candidates.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        candidates[right]
            .tokens
            .len()
            .cmp(&candidates[left].tokens.len())
            .then_with(|| candidates[right].area.total_cmp(&candidates[left].area))
            .then_with(|| candidates[left].layer.cmp(&candidates[right].layer))
    });
    let mut removed = BTreeSet::new();
    let mut merges = Vec::new();
    for (position, &keeper_index) in order.iter().enumerate() {
        let keeper = &candidates[keeper_index];
        if removed.contains(&keeper.layer) {
            continue;
        }
        for &candidate_index in &order[position + 1..] {
            let candidate = &candidates[candidate_index];
            if removed.contains(&candidate.layer) || keeper.tokens.len() < candidate.tokens.len() {
                continue;
            }
            let overlap = overlap_over_smaller(keeper.bounds, candidate.bounds);
            if overlap < MERGE_OVER_SMALLER_MINIMUM {
                continue;
            }
            let token_containment = token_sequence_containment(&keeper.tokens, &candidate.tokens);
            let exact = keeper.tokens == candidate.tokens && !keeper.tokens.is_empty();
            let fuzzy = candidate.tokens.len() >= MINIMUM_FUZZY_TOKENS
                && token_containment >= TOKEN_CONTAINMENT_MINIMUM;
            if !exact && !fuzzy {
                continue;
            }
            removed.insert(candidate.layer);
            merges.push(SourceRegionMerge {
                kept_element_id: keeper.layer,
                kept_content_id: keeper.content,
                kept_source_region_id: keeper.region,
                removed_element_id: candidate.layer,
                removed_content_id: candidate.content,
                removed_source_region_id: candidate.region,
                kept_source_text: keeper.source.text.value.clone(),
                removed_source_text: candidate.source.text.value.clone(),
                source_region_overlap_over_smaller: overlap,
                source_token_sequence_containment: token_containment,
                reason: "near-coincident source region repeats a token subsequence of the richer retained OCR source",
            });
        }
    }
    merges
}

fn source_tokens(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_lowercase())
        .collect()
}

fn token_sequence_containment(richer: &[String], subset: &[String]) -> f64 {
    if subset.is_empty() {
        return 0.0;
    }
    let mut previous = vec![0usize; subset.len() + 1];
    for richer_token in richer {
        let mut current = vec![0usize; subset.len() + 1];
        for (index, subset_token) in subset.iter().enumerate() {
            current[index + 1] = if richer_token == subset_token {
                previous[index] + 1
            } else {
                current[index].max(previous[index + 1])
            };
        }
        previous = current;
    }
    previous[subset.len()] as f64 / subset.len() as f64
}

fn overlap_over_smaller(left: [f64; 4], right: [f64; 4]) -> f64 {
    let intersection = (left[2].min(right[2]) - left[0].max(right[0])).max(0.0)
        * (left[3].min(right[3]) - left[1].max(right[1])).max(0.0);
    let smaller = ((left[2] - left[0]) * (left[3] - left[1]))
        .min((right[2] - right[0]) * (right[3] - right[1]));
    if smaller > 0.0 {
        intersection / smaller
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use koharu_scene::{
        At, Authored, DetectionAnalysis, DetectionLabel, Geometry, LanguageTag, Origin, PageDraft,
        Region, SourceText, TextLayout, TextRegion, Typography,
    };

    use super::*;

    fn candidate(bounds: [f64; 4], source: &str) -> Candidate {
        Candidate {
            layer: EntityId::new(),
            content: EntityId::new(),
            region: EntityId::new(),
            bounds,
            area: (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]),
            source: SourceText {
                text: Authored::user(source.to_owned()),
                language: None,
            },
            tokens: source_tokens(source),
            typography: Typography {
                origin: Origin::User,
                preferred_font: None,
                font_weight: None,
                font_style: None,
                size: None,
                auto_fit: true,
                color: None,
                stroke_color: None,
                stroke_width: None,
                alignment: None,
                writing_mode: None,
                extensions: Default::default(),
            },
            target_region: None,
            target_bounds: None,
            verified_vertical_japanese_dialogue: false,
        }
    }

    fn verified_dialogue_candidate(bounds: [f64; 4], source: &str, target: EntityId) -> Candidate {
        let mut candidate = candidate(bounds, source);
        candidate.target_region = Some(target);
        candidate.target_bounds = Some([0.0, 0.0, 200.0, 240.0]);
        candidate.verified_vertical_japanese_dialogue = true;
        candidate
    }

    #[test]
    fn nested_ocr_suffix_is_merged_into_the_richer_source() {
        let richer = candidate(
            [95.144, 1436.640, 562.593, 1933.640],
            "...MY APOLOGIES. (O MAMA IS SO FAT, SHE DIED.",
        );
        let duplicate = candidate(
            [92.386, 1692.906, 485.375, 1941.406],
            "YO MAMA IS SO FAT, SHE DIED.",
        );
        let richer_id = richer.layer;
        let duplicate_id = duplicate.layer;
        let merges = justified_merges(&[richer, duplicate]);

        assert_eq!(merges.len(), 1);
        assert_eq!(merges[0].kept_element_id, richer_id);
        assert_eq!(merges[0].removed_element_id, duplicate_id);
        assert!(merges[0].source_region_overlap_over_smaller > 0.96);
        assert!(merges[0].source_token_sequence_containment > 0.85);
        assert!(merges[0].kept_source_text.starts_with("...MY APOLOGIES"));
    }

    #[test]
    fn overlapping_distinct_dialogue_is_not_merged() {
        let first = candidate([0.0, 0.0, 100.0, 100.0], "PLEASE WAIT FOR ME");
        let second = candidate([2.0, 2.0, 98.0, 98.0], "I WILL BE RIGHT BACK");
        assert!(justified_merges(&[first, second]).is_empty());
    }

    #[test]
    fn verified_vertical_members_share_one_ordered_korean_dialogue_group() {
        let target = EntityId::new();
        let right = verified_dialogue_candidate([130.0, 20.0, 160.0, 90.0], "右", target);
        let right_lower = verified_dialogue_candidate([130.0, 100.0, 160.0, 180.0], "下", target);
        let left = verified_dialogue_candidate([70.0, 30.0, 100.0, 170.0], "左", target);
        let groups = logical_dialogue_groups(&[left, right_lower, right], Language::Korean);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].target_region_id, target);
        assert_eq!(groups[0].logical_source_text, "右\n下\n左");
        assert_eq!(groups[0].members.len(), 3);
        assert_eq!(
            groups[0]
                .members
                .iter()
                .map(|member| member.ordinal)
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
    }

    #[test]
    fn dialogue_groups_never_cross_targets_or_include_unverified_free_text() {
        let first_target = EntityId::new();
        let second_target = EntityId::new();
        let first = verified_dialogue_candidate([130.0, 20.0, 160.0, 90.0], "一", first_target);
        let second = verified_dialogue_candidate([70.0, 20.0, 100.0, 90.0], "二", second_target);
        let free_text = candidate([10.0, 20.0, 40.0, 90.0], "効果音");

        assert!(logical_dialogue_groups(&[first, second, free_text], Language::Korean).is_empty());
    }

    #[tokio::test]
    async fn preprocessing_commits_before_inpainting_establishes_its_base() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut ids = Vec::new();
        let generation = Generation {
            producer: ProducerId::new(detection::PRODUCER).unwrap(),
            model: Some("detector".to_owned()),
            confidence: None,
        };
        let mut setup = session.snapshot().edit_as(generation.clone());
        let page = setup
            .add_page(PageDraft::new("page", 800.0, 800.0), At::End)
            .unwrap();
        for (geometry, text) in [
            (
                Geometry::rectangle(100.0, 100.0, 500.0, 500.0),
                "MY APOLOGIES YO MAMA IS SO FAT SHE DIED",
            ),
            (
                Geometry::rectangle(105.0, 300.0, 400.0, 295.0),
                "YO MAMA IS SO FAT SHE DIED",
            ),
        ] {
            let region = setup.add_entity(page, At::End).unwrap();
            setup.set(region, &geometry).unwrap();
            setup
                .set(
                    region,
                    &Region {
                        origin: Origin::Generated(generation.clone()),
                        kind: TextRegion::kind(),
                        label: Some("text".to_owned()),
                    },
                )
                .unwrap();
            setup
                .set(
                    region,
                    &DetectionAnalysis {
                        origin: Origin::Generated(generation.clone()),
                        labels: vec![DetectionLabel {
                            kind: TextRegion::kind(),
                            confidence: 0.9,
                        }],
                    },
                )
                .unwrap();
            let content = setup.add_text_content(page, At::End).unwrap();
            let layer = setup
                .add_text_layer(
                    page,
                    At::End,
                    content,
                    &TextLayout {
                        origin: Origin::Generated(generation.clone()),
                        kind: TextLayoutKind::Paragraph,
                    },
                )
                .unwrap();
            setup
                .set(
                    layer,
                    &Typography {
                        origin: Origin::Generated(generation.clone()),
                        preferred_font: None,
                        font_weight: None,
                        font_style: None,
                        size: None,
                        auto_fit: true,
                        color: None,
                        stroke_color: None,
                        stroke_width: None,
                        alignment: None,
                        writing_mode: Some(WritingMode::Vertical),
                        extensions: Default::default(),
                    },
                )
                .unwrap();
            setup.relate::<RecognizedFrom>(content, region).unwrap();
            setup.relate::<koharu_scene::FitsTo>(layer, region).unwrap();
            setup
                .set(
                    content,
                    &SourceText {
                        text: Authored::generated(text.to_owned(), generation.clone()),
                        language: Some(LanguageTag::new("en-US").unwrap()),
                    },
                )
                .unwrap();
            ids.push((layer, content, region));
        }
        let snapshot = session
            .commit(setup.finish().unwrap())
            .await
            .unwrap()
            .snapshot;
        let stale_inpainting_patch = snapshot
            .patch(|edit| {
                edit.add_entity(page, At::Start)?;
                Ok(())
            })
            .unwrap();
        let (patch, report) =
            prepare_translation_inputs(&snapshot, page, Language::Korean).unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;

        assert_eq!(report.merges.len(), 1);
        assert_eq!(report.style_adjustments.len(), 1);
        assert!(snapshot.entity(ids[0].0).is_ok());
        assert!(snapshot.entity(ids[1].0).is_err());
        assert_eq!(
            snapshot
                .component::<SourceText>(ids[0].1)
                .unwrap()
                .unwrap()
                .text
                .value,
            "MY APOLOGIES YO MAMA IS SO FAT SHE DIED"
        );
        let typography = snapshot.component::<Typography>(ids[0].0).unwrap().unwrap();
        assert_eq!(typography.size, Some(300.0));
        assert_eq!(typography.writing_mode, Some(WritingMode::Horizontal));

        assert!(stale_inpainting_patch.rebase_on(&snapshot).is_err());
        let current_inpainting_patch = snapshot
            .patch(|edit| {
                edit.add_entity(page, At::Start)?;
                Ok(())
            })
            .unwrap();
        assert!(current_inpainting_patch.rebase_on(&snapshot).is_ok());
    }
}
