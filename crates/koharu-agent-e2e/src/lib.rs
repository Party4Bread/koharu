mod acceptance;
mod free_dialogue;
mod host;
mod page_translation;
mod placement;
mod repair;
mod review;
mod sfx;
mod ui_panel;

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context as _, Result, bail};
use clap::{Parser, ValueEnum};
use koharu_agent::{
    Agent, Codex, Control, DEFAULT_PROVIDER_MAX_ATTEMPTS, DEFAULT_PROVIDER_RETRY_BASE_DELAY,
    DEFAULT_PROVIDER_RETRY_MAX_DELAY, ProviderRetryPolicy, RunId, TraceLocation,
};
use koharu_translator::Language;
use serde::Serialize;

use host::HarnessHost;

pub use acceptance::{AcceptanceRecord, QualityThresholds};
pub use host::PipelineTelemetry;
pub use repair::{
    DeterministicRepairFailure, DeterministicRepairPlan, RepairField, RepairNextAction,
};
pub use review::VisualReviewRecord;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Run Koharu Agent end-to-end without a desktop or webview"
)]
pub struct Arguments {
    #[arg(long = "input", required = true, value_name = "ABSOLUTE_PAGE_PATH")]
    pub inputs: Vec<PathBuf>,

    #[arg(long, value_name = "LANGUAGE")]
    pub source_language: Language,

    #[arg(long, value_name = "LANGUAGE")]
    pub target_language: Language,

    #[arg(long, value_name = "EXISTING_ABSOLUTE_DIRECTORY")]
    pub output_directory: PathBuf,

    /// Parent directory in which a run-specific visual/semantic review bundle is created.
    #[arg(long, value_name = "EXISTING_ABSOLUTE_DIRECTORY")]
    pub review_directory: PathBuf,

    /// Judge executable; receives the bundle manifest path and returns decision JSON.
    #[arg(long, value_name = "ABSOLUTE_EXECUTABLE_PATH")]
    pub visual_review_command: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Png)]
    pub output_format: OutputFormat,

    #[arg(long, value_name = "ABSOLUTE_JSONL_PATH")]
    pub trace: PathBuf,

    #[arg(long, value_name = "PROMPT")]
    pub prompt: String,

    /// Total request attempts for transient provider overload, rate-limit, server, or connection failures.
    #[arg(
        long,
        default_value_t = DEFAULT_PROVIDER_MAX_ATTEMPTS,
        value_name = "COUNT"
    )]
    pub provider_max_attempts: u32,

    /// Initial transient-provider retry delay; later delays grow exponentially with jitter.
    #[arg(
        long,
        default_value_t = DEFAULT_PROVIDER_RETRY_BASE_DELAY.as_millis() as u64,
        value_name = "MILLISECONDS"
    )]
    pub provider_retry_base_delay_ms: u64,

    /// Maximum delay between transient provider retries.
    #[arg(
        long,
        default_value_t = DEFAULT_PROVIDER_RETRY_MAX_DELAY.as_millis() as u64,
        value_name = "MILLISECONDS"
    )]
    pub provider_retry_max_delay_ms: u64,

    #[command(flatten)]
    pub quality_thresholds: QualityThresholds,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    Png,
    Psd,
}

#[derive(Debug, Serialize)]
pub struct Summary {
    pub ok: bool,
    pub run_id: RunId,
    pub input_paths: Vec<String>,
    pub source_language: String,
    pub target_language: String,
    pub output_format: OutputFormat,
    pub output_paths: Vec<String>,
    pub trace_path: String,
    pub agent_message: String,
    pub acceptance: AcceptanceRecord,
    pub visual_review: VisualReviewRecord,
    pub pipeline_telemetry: PipelineTelemetry,
    pub project_disposed: bool,
}

pub async fn execute(arguments: Arguments) -> Result<Summary> {
    let inputs = validate_inputs(arguments.inputs)?;
    let output_directory = existing_absolute_directory(&arguments.output_directory, "output")?;
    let review_directory = existing_absolute_directory(&arguments.review_directory, "review")?;
    let visual_review_command = arguments
        .visual_review_command
        .as_deref()
        .map(existing_absolute_file)
        .transpose()?;
    arguments.quality_thresholds.validate()?;
    let provider_retry = ProviderRetryPolicy::new(
        arguments.provider_max_attempts,
        Duration::from_millis(arguments.provider_retry_base_delay_ms),
        Duration::from_millis(arguments.provider_retry_max_delay_ms),
    )?;
    let prompt = arguments.prompt.trim();
    if prompt.is_empty() {
        bail!("agent prompt cannot be empty");
    }

    let run = RunId::new();
    let review_bundle_directory = review_directory.join(run.to_string());
    let retained_review_directory = review_bundle_directory.clone();
    let trace = TraceLocation::explicit(run, &arguments.trace)?;
    if Path::new(&trace.path).exists() {
        bail!("agent trace path already exists: {}", trace.path);
    }

    let codex = Codex::new()?;
    if codex.account()?.is_none() {
        bail!("Codex is not signed in; sign in with Koharu before running the harness");
    }

    let host = HarnessHost::create(
        inputs.clone(),
        arguments.source_language,
        arguments.target_language,
        output_directory,
        arguments.output_format,
        arguments.quality_thresholds.clone(),
        review_bundle_directory,
        visual_review_command,
    )
    .await?;
    let project_path = host.project_path().to_owned();
    let output_paths = host.output_paths();
    let agent = Agent::new(codex, host.clone())?;
    let harness_prompt = harness_prompt(
        prompt,
        arguments.source_language,
        arguments.target_language,
        arguments.output_format,
    );
    let result = agent
        .run(
            run,
            harness_prompt,
            Control::default(),
            provider_retry,
            trace.clone(),
            |_| {},
        )
        .await
        .with_context(|| {
            format!(
                "agent run failed; trace retained at {}; any review artifacts remain under {}",
                trace.path,
                retained_review_directory.display(),
            )
        })?;
    let outputs = output_paths.lock().clone();
    if !host.pipeline_completed() {
        bail!("agent completed without running the Koharu pipeline");
    }
    if outputs.len() != inputs.len() {
        bail!(
            "agent exported {} pages, but {} input pages were imported",
            outputs.len(),
            inputs.len()
        );
    }
    for output in &outputs {
        if !output.is_file() {
            bail!("agent reported a missing export: {}", output.display());
        }
    }
    let acceptance = host
        .acceptance_record()
        .context("the harness completed without an acceptance record")?;
    if !acceptance.accepted {
        bail!("the exported project did not satisfy acceptance criteria");
    }
    let visual_review = host
        .visual_review_record()
        .context("the harness completed without a visual-review record")?;
    if !visual_review.accepted() {
        bail!("the exported project did not pass the required visual/semantic review");
    }
    let pipeline_telemetry = host
        .pipeline_telemetry()
        .context("the harness completed without pipeline telemetry")?;
    drop(agent);
    drop(host);

    let input_paths = paths_to_strings(&inputs)?;
    let output_paths = paths_to_strings(&outputs)?;
    let project_disposed = !project_path.exists();
    if !project_disposed {
        bail!(
            "disposable project was not removed: {}",
            project_path.display()
        );
    }
    Ok(Summary {
        ok: true,
        run_id: run,
        input_paths,
        source_language: arguments.source_language.tag().to_owned(),
        target_language: arguments.target_language.tag().to_owned(),
        output_format: arguments.output_format,
        output_paths,
        trace_path: trace.path,
        agent_message: result.message,
        acceptance,
        visual_review,
        pipeline_telemetry,
        project_disposed,
    })
}

impl OutputFormat {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Psd => "psd",
        }
    }
}

#[cfg(test)]
pub(crate) const INITIAL_CONTEXT_BUDGET_ESTIMATED_TOKENS: usize = 16_384;

pub(crate) fn harness_prompt(
    request: &str,
    source_language: Language,
    target_language: Language,
    output_format: OutputFormat,
) -> String {
    format!(
        "{request}\n\nHeadless harness contract: process every imported page from {source_language} to {target_language}. Follow the phase-gated tools until export; unavailable tools are not authorized. Run source analysis, then inspect each original dossier and source-debug overlay. Original pixels are authoritative; OCR is only an aid. Classify decorative SFX only from fresh visual-form, page-function, role, geometry, and container evidence, never from literal tokens or script vocabulary. Legible SFX remain required for translation with detected typography preserved; only positively evidenced difficult SFX may skip. Bubble dialogue, captions, UI, container-associated text, and general free text retain their existing gates. The host binds exact positively proven UI text to its detector-backed finite panel safe interior using the recorded role, source identity, containment, and provenance; use verify_ui_panel_anchor only if it remains dynamically exposed for genuinely unbound evidence, never infer or select a nearest panel. Translate all remaining required content with run_pipeline. Any adjacent free-dialogue anchor remains limited to required compact no-container source text with the enforced role, writing-mode, source-pixel proximity/structure/contrast/room, reading-order, and attribution gates. Only when every adjacent patch fails may the host-recorded source-bound fallback preview the geometry-proven native-vertical form for a target of at most two visible graphemes; it still must pass the complete deterministic rerender before commit.\n\nFor each translated page refresh the four artifacts in order: inspect_source_evidence + view_page_source_debug, then review_page_translation + view_page_debug. Compare source pixels/crops with the complete ordered target render. review_pages determines acceptance and the binding first repair. Judge source accuracy, meaning, natural target language, reading order, completeness/duplicates, typography/layout, every skip, and any moved dialogue's speaker/reaction attribution; deterministic geometry is not semantic evidence. In in-loop review, submit the exact page/revision and four digests with all seven judgments. Recheck compacted text against every bound original group-member crop.\n\nUse only the repair tool, element/group, fields, and host-issued preview exposed for the active deterministic plan; never submit raw geometry. Refresh all four artifacts and review again after each mutation. Every distinct deterministic plan action receives one preview and global-validation attempt; the host stops rather than replaying the same operation on the same target. Do not invent wording without page and original-pixel evidence. Export every page as {} only after fresh deterministic acceptance and the required current-revision semantic decision. Do not reply before the gated workflow completes.",
        output_format.as_str(),
    )
}

fn validate_inputs(inputs: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    if inputs.is_empty() {
        bail!("at least one input page path is required");
    }
    inputs
        .into_iter()
        .map(|path| {
            if !path.is_absolute() {
                bail!("input page path must be absolute: {}", path.display());
            }
            let path = path
                .canonicalize()
                .with_context(|| format!("failed to resolve input page {}", path.display()))?;
            if !path.is_file() {
                bail!("input page path is not a file: {}", path.display());
            }
            Ok(path)
        })
        .collect()
}

fn existing_absolute_directory(path: &Path, label: &str) -> Result<PathBuf> {
    if !path.is_absolute() {
        bail!("{label} directory must be absolute: {}", path.display());
    }
    let path = path
        .canonicalize()
        .with_context(|| format!("failed to resolve {label} directory {}", path.display()))?;
    if !path.is_dir() {
        bail!("{label} path is not a directory: {}", path.display());
    }
    Ok(path)
}

fn existing_absolute_file(path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        bail!("visual-review command must be absolute: {}", path.display());
    }
    let path = path
        .canonicalize()
        .with_context(|| format!("failed to resolve visual-review command {}", path.display()))?;
    if !path.is_file() {
        bail!("visual-review command is not a file: {}", path.display());
    }
    Ok(path)
}

fn paths_to_strings(paths: &[PathBuf]) -> Result<Vec<String>> {
    paths
        .iter()
        .map(|path| {
            path.to_str()
                .context("a result path is not valid UTF-8")
                .map(ToOwned::to_owned)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use clap::Parser as _;
    use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};

    use super::*;

    #[test]
    fn command_accepts_repeated_explicit_inputs() {
        let arguments = Arguments::try_parse_from([
            "koharu-agent-e2e",
            "--input",
            "/tmp/one.png",
            "--input",
            "/tmp/two.png",
            "--target-language",
            "ko",
            "--source-language",
            "en",
            "--output-directory",
            "/tmp",
            "--output-format",
            "psd",
            "--review-directory",
            "/tmp",
            "--min-rendered-font-size-px",
            "18",
            "--min-text-safe-padding-px",
            "6.5",
            "--trace",
            "/tmp/run.jsonl",
            "--prompt",
            "translate and export",
        ])
        .unwrap();
        assert_eq!(arguments.inputs.len(), 2);
        assert_eq!(arguments.source_language, Language::English);
        assert_eq!(arguments.target_language, Language::Korean);
        assert_eq!(arguments.output_format, OutputFormat::Psd);
        assert_eq!(arguments.quality_thresholds.min_rendered_font_size_px, 18.0);
        assert_eq!(arguments.quality_thresholds.min_text_safe_padding_px, 6.5);
        assert_eq!(arguments.provider_max_attempts, 4);
        assert_eq!(arguments.provider_retry_base_delay_ms, 1_000);
        assert_eq!(arguments.provider_retry_max_delay_ms, 30_000);
    }

    #[test]
    fn command_accepts_explicit_provider_retry_limits() {
        let arguments = Arguments::try_parse_from([
            "koharu-agent-e2e",
            "--input",
            "/tmp/page.png",
            "--target-language",
            "ko",
            "--source-language",
            "en",
            "--output-directory",
            "/tmp",
            "--review-directory",
            "/tmp",
            "--trace",
            "/tmp/run.jsonl",
            "--prompt",
            "translate and export",
            "--provider-max-attempts",
            "6",
            "--provider-retry-base-delay-ms",
            "250",
            "--provider-retry-max-delay-ms",
            "8000",
        ])
        .unwrap();

        assert_eq!(arguments.provider_max_attempts, 6);
        assert_eq!(arguments.provider_retry_base_delay_ms, 250);
        assert_eq!(arguments.provider_retry_max_delay_ms, 8_000);
    }

    #[test]
    fn paths_must_be_absolute_files_and_existing_directories() {
        assert!(validate_inputs(vec![PathBuf::from("page.png")]).is_err());
        assert!(existing_absolute_directory(Path::new("exports"), "output").is_err());
    }

    #[tokio::test]
    async fn disposable_project_preserves_two_ordered_png_pages_and_is_removed_on_drop() {
        let fixture = tempfile::tempdir().unwrap();
        let inputs = [
            ("page-b.png", 2, 3, Rgba([1, 2, 3, 255])),
            ("page-a.png", 4, 1, Rgba([4, 5, 6, 255])),
        ]
        .map(|(name, width, height, pixel)| {
            let input = fixture.path().join(name);
            let mut bytes = Cursor::new(Vec::new());
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(width, height, pixel))
                .write_to(&mut bytes, ImageFormat::Png)
                .unwrap();
            std::fs::write(&input, bytes.into_inner()).unwrap();
            input
        });
        let project = host::DisposableProject::create(inputs.to_vec())
            .await
            .unwrap();
        let path = project.path().to_owned();
        assert!(path.is_dir());
        let snapshot = project.session().lock().await.snapshot();
        let pages = snapshot
            .pages()
            .map(|page| {
                let page = page.page().unwrap();
                (page.label, page.width, page.height)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            pages,
            [
                ("page-b.png".to_owned(), 2.0, 3.0),
                ("page-a.png".to_owned(), 4.0, 1.0)
            ]
        );
        assert_eq!(
            project
                .originals()
                .iter()
                .map(|page| (page.label.as_str(), page.media_type.as_str()))
                .collect::<Vec<_>>(),
            [("page-b.png", "image/png"), ("page-a.png", "image/png")]
        );
        drop(project);
        assert!(!path.exists());
    }
}
