use std::collections::BTreeSet;

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use koharu_scene::{Authored, LanguageTag, Origin, SourceText, Translation};
use koharu_translator::{TranslationRequest, Translator};

use crate::TranslationConfig;

use super::{StageInput, StageProcessor, finish, generation};

const PRODUCER: &str = "dev.koharu.pipeline.translation";
const SKIPPED_DIFFICULT_SFX_ROLE: &str = "dev.koharu.text.skipped-difficult-sfx";

pub(super) struct Processor {
    config: TranslationConfig,
    translator: Translator,
}

impl Processor {
    pub(super) fn new(config: TranslationConfig, translator: Translator) -> Self {
        Self { config, translator }
    }

    pub(super) fn target_language(&self) -> koharu_translator::Language {
        self.config.target_language
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        Translator::model(&self.config.model)
    }

    fn unload(&self) -> bool {
        self.translator.unload()
    }

    async fn load(&self) -> Result<()> {
        self.translator.load_model(&self.config.model).await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let mut targets = Vec::new();
        if let Some(group) = input.scene.page(input.page)?.text_group()? {
            let layers = group.text_layers()?.collect::<Vec<_>>();
            let logical_dialogues = layers
                .iter()
                .filter_map(|layer| {
                    let content = layer.content().ok()?;
                    content
                        .logical_dialogue()
                        .ok()?
                        .map(|dialogue| (content.id(), dialogue))
                })
                .collect::<Vec<_>>();
            let grouped_contents = logical_dialogues
                .iter()
                .flat_map(|(_, dialogue)| dialogue.members.iter().map(|member| member.content_id))
                .collect::<BTreeSet<_>>();
            for layer in layers {
                if !input.contains_entity(layer.id())? {
                    continue;
                }
                let content = layer.content()?;
                if content
                    .role()?
                    .is_some_and(|role| role.role == SKIPPED_DIFFICULT_SFX_ROLE)
                {
                    continue;
                }
                if let Some((_, dialogue)) = logical_dialogues
                    .iter()
                    .find(|(primary_content, _)| *primary_content == content.id())
                {
                    let source = dialogue
                        .members
                        .iter()
                        .map(|member| {
                            input
                                .scene
                                .component::<SourceText>(member.content_id)?
                                .context("logical dialogue member is missing source text")
                                .map(|source| source.text.value)
                        })
                        .collect::<Result<Vec<_>>>()?
                        .join("\n");
                    if !source.trim().is_empty() {
                        targets.push((content.id(), source));
                    }
                    continue;
                }
                if grouped_contents.contains(&content.id()) {
                    continue;
                }
                let Some(source) = content.source()? else {
                    continue;
                };
                if !source.text.value.trim().is_empty() {
                    targets.push((content.id(), source.text.value));
                }
            }
        }
        let mut request = translation_request(
            targets.iter().map(|(_, source)| source.clone()),
            self.config.target_language,
            input.source_language(),
        );
        if let Some(instructions) = self.config.instructions.as_deref() {
            request = request.with_instructions(instructions);
        }
        if Translator::supports_vision(&self.config.model, &self.config.generation)
            && let Some(image) = input.images.get(&input.scene, input.page, "source").await?
        {
            request = request.with_image(image);
        }
        let (provider, translated) = self
            .translator
            .translate(&self.config.model, self.config.generation, request)
            .await?;
        let language = LanguageTag::new(self.config.target_language.tag())?;
        let generated = generation(PRODUCER, provider)?;
        let mut edit = input.scene.edit_as(generated.clone());
        for (entity, _) in &targets {
            edit.observe::<SourceText>(*entity)?;
            edit.observe::<Translation>(*entity)?;
        }
        for ((entity, source), text) in targets.into_iter().zip(translated) {
            if input
                .scene
                .component::<Translation>(entity)?
                .is_some_and(|value| matches!(value.text.origin, Origin::User))
            {
                continue;
            }
            let text = if source.trim() == "\u{2026}" {
                "\u{2026}".to_owned()
            } else {
                text
            };
            edit.set(
                entity,
                &Translation {
                    text: Authored::generated(text, generated.clone()),
                    language: Some(language.clone()),
                },
            )?;
        }
        finish(edit)
    }
}

fn translation_request(
    segments: impl IntoIterator<Item = String>,
    target_language: koharu_translator::Language,
    source_language: Option<koharu_translator::Language>,
) -> TranslationRequest {
    let request = TranslationRequest::new(segments, target_language);
    match source_language {
        Some(language) => request.with_source_language(language),
        None => request,
    }
}

#[cfg(test)]
mod tests {
    use koharu_translator::Language;

    use super::translation_request;

    #[test]
    fn requested_source_language_is_forwarded_to_translation_provider() {
        let request = translation_request(
            ["hello".to_owned()],
            Language::Korean,
            Some(Language::English),
        );
        assert_eq!(request.source_language, Some(Language::English));
        assert_eq!(request.target_language, Language::Korean);
    }
}
