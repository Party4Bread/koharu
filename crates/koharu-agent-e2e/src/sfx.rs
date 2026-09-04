use koharu_scene::{EntityId, Revision, Typography};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::acceptance::SemanticText;

pub(crate) const SKIPPED_DIFFICULT_SFX_ROLE: &str = "dev.koharu.text.skipped-difficult-sfx";
pub(crate) const DECORATIVE_SFX_ROLE: &str = "dev.koharu.text.sfx";
pub(crate) const FREE_TEXT_ROLE: &str = "dev.koharu.text.free-text";
pub(crate) const MINIMUM_SFX_CLASSIFICATION_CONFIDENCE: f32 = 0.90;
pub(crate) const DECORATIVE_SFX_DECISION_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct DecorativeSfxDecision {
    pub schema_version: u32,
    pub disposition: DecorativeSfxDisposition,
    pub review_state: &'static str,
    pub evidence_revision: Revision,
    pub decision_revision: Revision,
    pub page_id: EntityId,
    pub original_ordinal: usize,
    pub element_id: EntityId,
    pub content_id: EntityId,
    pub source_region_id: EntityId,
    pub source_ocr: SemanticText,
    pub source_typography: Typography,
    pub source_crop_blake3: String,
    pub source_debug_label: String,
    pub classifier: SfxClassifier,
    pub evidence: DecorativeSfxEvidence,
    pub confidence: f32,
    pub rationale: String,
    pub target_translation_owner: Option<EntityId>,
    pub target_render_owner: Option<EntityId>,
}

impl DecorativeSfxDecision {
    pub(crate) fn is_skipped(&self) -> bool {
        self.disposition == DecorativeSfxDisposition::SkipDifficult
    }

    pub(crate) fn is_translated(&self) -> bool {
        self.disposition == DecorativeSfxDisposition::Translate
    }

    pub(crate) fn requires_translation(&self) -> bool {
        self.disposition != DecorativeSfxDisposition::SkipDifficult
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DecorativeSfxDisposition {
    Translate,
    SkipDifficult,
    RetainRequired,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SfxClassifier {
    pub kind: &'static str,
    pub configured_model: Option<String>,
    pub tool_call_id: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DecorativeSfxEvidence {
    pub decorative_visual_form: String,
    pub sound_effect_page_function: String,
    pub legibility_and_translation_value: String,
    pub exclusion_of_dialogue_caption_ui_and_general_free_text: String,
}
