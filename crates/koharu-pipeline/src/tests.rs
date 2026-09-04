use super::*;

use std::sync::{Arc, Mutex};

use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, Authored, BubbleRegion, DetectionAnalysis,
    DetectionLabel, FlowsIn, Generation, Geometry, Inside, LanguageTag, OcrAnalysis, Origin,
    PageDraft, ProducerId, RasterLayer, RasterLayerKind, RecognizedFrom, RegionSpec, SourceText,
    TextDirection, TextLayout, TextLayoutKind, TextRegion, TextRole, Translation, Typography,
    WritingMode,
};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

#[test]
fn configuration_ignores_unknown_fields() {
    let config = toml::from_str::<PipelineConfig>("legacy_limit = 1").unwrap();
    assert_eq!(config, PipelineConfig::default());
}

#[tokio::test]
async fn stop_is_a_successful_partial_result() {
    let pipeline = pipeline(Default::default());
    let stop = StopToken::default();
    stop.stop();
    let request = Request {
        stop,
        ..Request::default()
    };
    let mut committer = RejectCommitter;

    let report = pipeline
        .execute(
            koharu_scene::Session::memory().await.unwrap().snapshot(),
            request,
            &mut committer,
        )
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Stopped);
    assert_eq!(report.completed, 0);
}

#[tokio::test]
async fn stop_after_a_page_keeps_completed_progress() {
    let translation = TranslationConfig {
        model: koharu_translator::ModelSelection {
            provider: koharu_translator::Provider::OpenAi,
            model: Some("gpt-5.6-luna".to_owned()),
            quantization: None,
            vision: true,
            reasoning: true,
        },
        ..Default::default()
    };
    let pipeline = pipeline(translation);
    let mut session = koharu_scene::Session::memory().await.unwrap();
    let patch = session
        .snapshot()
        .patch(|edit| {
            edit.add_page(
                koharu_scene::PageDraft::new("one", 1.0, 1.0),
                koharu_scene::At::End,
            )?;
            edit.add_page(
                koharu_scene::PageDraft::new("two", 1.0, 1.0),
                koharu_scene::At::End,
            )?;
            Ok(())
        })
        .unwrap();
    session.commit(patch).await.unwrap();
    let stop = StopToken::default();
    let progress_stop = stop.clone();
    let request = Request {
        operation: Operation::Only {
            stage: Stage::Translation,
        },
        stop,
        progress: Some(std::sync::Arc::new(move |event| {
            if matches!(event, Progress::NoOp { .. }) {
                progress_stop.stop();
            }
        })),
        ..Request::default()
    };
    let mut committer = RejectCommitter;

    let report = pipeline
        .execute(session.snapshot(), request, &mut committer)
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Stopped);
    assert_eq!(report.completed, 1);
    assert_eq!(report.total, 2);
}

struct RejectCommitter;

struct SessionCommitter<'a> {
    session: &'a mut koharu_scene::Session,
    commits: usize,
}

fn pipeline(translation: TranslationConfig) -> Pipeline {
    pipeline_with_providers(translation, koharu_translator::ProvidersConfig::default())
}

fn pipeline_with_providers(
    translation: TranslationConfig,
    providers: koharu_translator::ProvidersConfig,
) -> Pipeline {
    let config = PipelineConfig {
        translation,
        ..PipelineConfig::default()
    };
    Pipeline::from_config(
        koharu_config::Config::memory(config),
        koharu_config::Config::memory(providers),
        koharu_ml::Device::cpu(),
    )
    .unwrap()
}

#[async_trait::async_trait]
impl Committer for RejectCommitter {
    async fn commit(&mut self, _output: StageOutput) -> anyhow::Result<koharu_scene::Snapshot> {
        anyhow::bail!("stopped execution must not commit")
    }
}

#[async_trait::async_trait]
impl Committer for SessionCommitter<'_> {
    async fn commit(&mut self, output: StageOutput) -> anyhow::Result<koharu_scene::Snapshot> {
        assert!(
            !output.patch.is_empty(),
            "no-op patches must not reach the committer"
        );
        self.commits += 1;
        Ok(self.session.commit(output.patch).await?.snapshot)
    }
}

fn test_asset() -> AssetInput {
    AssetInput::new(
        Arc::<[u8]>::from(&b"test image"[..]),
        "image/png",
        AssetMetadata {
            width: Some(1),
            height: Some(1),
            attributes: Default::default(),
        },
    )
}

fn add_analyzed_page(
    edit: &mut koharu_scene::Edit,
    label: &str,
    source: &str,
    skipped_difficult_sfx: bool,
) -> (koharu_scene::EntityId, koharu_scene::EntityId) {
    let page = edit
        .add_page(PageDraft::new(label, 100.0, 100.0), At::End)
        .unwrap();
    let content = edit.add_text_content(page, At::End).unwrap();
    edit.add_text_layer(
        page,
        At::End,
        content,
        &TextLayout {
            origin: Origin::User,
            kind: TextLayoutKind::Paragraph,
        },
    )
    .unwrap();
    edit.set(
        content,
        &SourceText {
            text: Authored::user(source.to_owned()),
            language: Some(LanguageTag::new("ja-JP").unwrap()),
        },
    )
    .unwrap();
    if skipped_difficult_sfx {
        edit.set(
            content,
            &TextRole {
                origin: Origin::User,
                role: "dev.koharu.text.skipped-difficult-sfx".to_owned(),
            },
        )
        .unwrap();
    }
    let cleanup = edit.add_entity(page, At::End).unwrap();
    edit.set(
        cleanup,
        &RasterLayer {
            origin: Origin::User,
            name: "Cleanup".to_owned(),
            kind: RasterLayerKind::Cleanup,
        },
    )
    .unwrap();
    edit.set_asset(cleanup, &AssetRole::new("source").unwrap(), test_asset())
        .unwrap();
    (page, content)
}

fn add_analyzed_dialogue_group(
    edit: &mut koharu_scene::Edit,
) -> (koharu_scene::EntityId, [koharu_scene::EntityId; 2]) {
    let generation = Generation {
        producer: ProducerId::new("dev.koharu.pipeline.detection").unwrap(),
        model: Some("test-detector".to_owned()),
        confidence: None,
    };
    let page = edit
        .add_page(PageDraft::new("grouped dialogue", 200.0, 240.0), At::End)
        .unwrap();
    let bubble = edit
        .add_analysis_region::<BubbleRegion>(
            page,
            At::End,
            &Geometry::rectangle(0.0, 0.0, 200.0, 240.0),
            Some("bubble".to_owned()),
        )
        .unwrap();
    let mut contents = Vec::new();
    for (x, source) in [(130.0, "右"), (70.0, "左")] {
        let region = edit
            .add_analysis_region::<TextRegion>(
                page,
                At::End,
                &Geometry::rectangle(x, 20.0, 30.0, 160.0),
                Some("text".to_owned()),
            )
            .unwrap();
        edit.set(
            region,
            &DetectionAnalysis {
                origin: Origin::Generated(generation.clone()),
                labels: vec![DetectionLabel {
                    kind: TextRegion::kind(),
                    confidence: 0.99,
                }],
            },
        )
        .unwrap();
        edit.set(
            region,
            &OcrAnalysis {
                origin: Origin::Generated(generation.clone()),
                direction: TextDirection::Vertical,
                confidence: Some(0.99),
                line_boundaries: Vec::new(),
            },
        )
        .unwrap();
        let content = edit.add_text_content(page, At::End).unwrap();
        let layer = edit
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
        edit.set(
            layer,
            &Typography {
                origin: Origin::Generated(generation.clone()),
                preferred_font: None,
                font_weight: None,
                font_style: None,
                size: Some(30.0),
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
        edit.set(
            content,
            &TextRole {
                origin: Origin::Generated(generation.clone()),
                role: "dev.koharu.text.dialogue".to_owned(),
            },
        )
        .unwrap();
        edit.set(
            content,
            &SourceText {
                text: Authored::generated(source.to_owned(), generation.clone()),
                language: Some(LanguageTag::new("ja-JP").unwrap()),
            },
        )
        .unwrap();
        edit.relate::<RecognizedFrom>(content, region).unwrap();
        edit.relate::<Inside>(region, bubble).unwrap();
        edit.relate::<FlowsIn>(layer, bubble).unwrap();
        contents.push(content);
    }
    let cleanup = edit.add_entity(page, At::End).unwrap();
    edit.set(
        cleanup,
        &RasterLayer {
            origin: Origin::User,
            name: "Cleanup".to_owned(),
            kind: RasterLayerKind::Cleanup,
        },
    )
    .unwrap();
    edit.set_asset(cleanup, &AssetRole::new("source").unwrap(), test_asset())
        .unwrap();
    (page, contents.try_into().unwrap())
}

async fn translation_server() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let read = stream.read(&mut buffer).await.unwrap();
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
            let Some(headers_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n")
            else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..headers_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .and_then(|value| value.trim().parse::<usize>().ok())
                })
                .unwrap_or(0);
            if request.len() >= headers_end + 4 + content_length {
                break;
            }
        }
        let body = r#"{"choices":[{"message":{"content":"{\"translations\":[{\"id\":0,\"text\":\"안녕하세요\"}]}"}}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body,
        );
        stream.write_all(response.as_bytes()).await.unwrap();
    });
    (format!("http://{address}/v1"), task)
}

#[tokio::test]
async fn analyzed_sources_complete_full_pipeline_with_a_valid_no_op_stage() {
    let (base_url, server) = translation_server().await;
    let providers = toml::from_str::<koharu_translator::ProvidersConfig>(&format!(
        "[openai-compatible]\nbase_url = \"{base_url}\"\n"
    ))
    .unwrap();
    let translation = TranslationConfig {
        model: koharu_translator::ModelSelection {
            provider: koharu_translator::Provider::OpenAiCompatible,
            model: Some("test-translator".to_owned()),
            quantization: None,
            vision: false,
            reasoning: false,
        },
        generation: koharu_translator::GenerationConfig {
            vision: Some(false),
            ..Default::default()
        },
        target_language: koharu_translator::Language::Korean,
        instructions: None,
    };
    let pipeline = pipeline_with_providers(translation, providers);
    let mut session = koharu_scene::Session::memory().await.unwrap();
    let mut analysis = session.snapshot().edit();
    let (skipped_page, skipped_content) =
        add_analyzed_page(&mut analysis, "skipped SFX", "ドン", true);
    let (required_page, required_content) =
        add_analyzed_page(&mut analysis, "required dialogue", "こんにちは", false);
    session.commit(analysis.finish().unwrap()).await.unwrap();

    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&events);
    let request = Request {
        operation: Operation::Stages {
            stages: vec![Stage::Translation, Stage::Inpainting],
        },
        source_language: Some(koharu_translator::Language::Japanese),
        progress: Some(Arc::new(move |event| captured.lock().unwrap().push(event))),
        ..Request::default()
    };
    let base = session.snapshot();
    let base_revision = base.revision();
    let mut committer = SessionCommitter {
        session: &mut session,
        commits: 0,
    };
    let report = pipeline
        .execute(base, request, &mut committer)
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Completed);
    assert_eq!((report.completed, report.total), (4, 4));
    assert_eq!(report.final_revision, base_revision.next().unwrap());
    assert_eq!(committer.commits, 1);
    server.await.unwrap();

    let snapshot = session.snapshot();
    assert!(
        snapshot
            .component::<Translation>(skipped_content)
            .unwrap()
            .is_none()
    );
    let required = snapshot
        .component::<Translation>(required_content)
        .unwrap()
        .unwrap();
    assert_eq!(required.text.value, "안녕하세요");
    assert_eq!(required.language.unwrap().as_str(), "ko-KR");

    let events = events.lock().unwrap();
    assert!(events.iter().any(|event| matches!(
        event,
        Progress::NoOp { page, stage: Stage::Translation, .. } if *page == skipped_page
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        Progress::Finished { page, stage: Stage::Translation, .. } if *page == required_page
    )));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                Progress::Skipped {
                    stage: Stage::Inpainting,
                    ..
                }
            ))
            .count(),
        2,
    );
}

#[tokio::test]
async fn repeated_source_analysis_preprocessing_no_op_does_not_abort_translation() {
    let (base_url, server) = translation_server().await;
    let providers = toml::from_str::<koharu_translator::ProvidersConfig>(&format!(
        "[openai-compatible]\nbase_url = \"{base_url}\"\n"
    ))
    .unwrap();
    let translation = TranslationConfig {
        model: koharu_translator::ModelSelection {
            provider: koharu_translator::Provider::OpenAiCompatible,
            model: Some("test-translator".to_owned()),
            quantization: None,
            vision: false,
            reasoning: false,
        },
        generation: koharu_translator::GenerationConfig {
            vision: Some(false),
            ..Default::default()
        },
        target_language: koharu_translator::Language::Korean,
        instructions: None,
    };
    let pipeline = pipeline_with_providers(translation, providers);
    let mut session = koharu_scene::Session::memory().await.unwrap();
    let mut analysis = session.snapshot().edit_as(Generation {
        producer: ProducerId::new("dev.koharu.pipeline.detection").unwrap(),
        model: Some("test-detector".to_owned()),
        confidence: None,
    });
    let (page, contents) = add_analyzed_dialogue_group(&mut analysis);
    session.commit(analysis.finish().unwrap()).await.unwrap();

    let (first_preprocessing, first_report) = prepare_translation_inputs(
        &session.snapshot(),
        page,
        koharu_translator::Language::Korean,
    )
    .unwrap();
    assert!(first_report.adjusted());
    assert!(!first_preprocessing.is_empty());
    session.commit(first_preprocessing).await.unwrap();
    let (repeated_preprocessing, repeated_report) = prepare_translation_inputs(
        &session.snapshot(),
        page,
        koharu_translator::Language::Korean,
    )
    .unwrap();
    assert!(repeated_report.adjusted());
    assert!(repeated_preprocessing.is_empty());

    let base = session.snapshot();
    let mut committer = SessionCommitter {
        session: &mut session,
        commits: 0,
    };
    let report = pipeline
        .execute(
            base,
            Request {
                operation: Operation::Stages {
                    stages: vec![Stage::Translation, Stage::Inpainting],
                },
                source_language: Some(koharu_translator::Language::Japanese),
                ..Request::default()
            },
            &mut committer,
        )
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Completed);
    assert_eq!(committer.commits, 1);
    server.await.unwrap();
    let snapshot = session.snapshot();
    assert!(
        snapshot
            .component::<Translation>(contents[0])
            .unwrap()
            .is_some()
    );
    assert!(
        snapshot
            .component::<Translation>(contents[1])
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn revision_advancement_remains_required_for_nonempty_stage_output() {
    let snapshot = koharu_scene::Session::memory().await.unwrap().snapshot();
    assert!(crate::execution::validate_commit(&snapshot, &snapshot).is_err());
    assert!(snapshot.edit().finish().unwrap().is_empty());
}

#[test]
fn operations_expand_to_the_supported_workflows() {
    assert_eq!(
        Operation::Through {
            stage: Stage::Translation,
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Ocr, Stage::Translation],
    );
    assert_eq!(
        Operation::Through {
            stage: Stage::Inpainting,
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Inpainting],
    );
    assert_eq!(
        Operation::Only {
            stage: Stage::Translation,
        }
        .stages()
        .unwrap(),
        vec![Stage::Translation],
    );
    assert_eq!(
        Operation::Stages {
            stages: vec![Stage::Translation, Stage::Detection, Stage::Translation],
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Translation],
    );
}
