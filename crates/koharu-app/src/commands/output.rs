use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context as _, Result, bail};
use koharu_rasterizer::Rasterizer;
use koharu_renderer::Renderer;
use koharu_scene::{AssetRole, EntityId, Snapshot};
use serde::Deserialize;
use specta::Type;
use tauri::{AppHandle, Cef, Manager as _, State, WebviewWindow, ipc::IpcResponse};

use super::{Error, project::CurrentProject};
use koharu_desktop::Desktop;

const THUMBNAIL_EDGE: u32 = 128;

#[derive(Type)]
#[specta(transparent)]
pub(crate) struct ThumbnailBytes(#[specta(type = Vec<u8>)] Vec<u8>);

impl IpcResponse for ThumbnailBytes {
    fn body(self) -> tauri::Result<tauri::ipc::InvokeResponseBody> {
        Ok(self.0.into())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Png,
    Psd,
}

impl From<ExportFormat> for koharu_desktop::ExportFormat {
    fn from(value: ExportFormat) -> Self {
        match value {
            ExportFormat::Png => Self::Png,
            ExportFormat::Psd => Self::Psd,
        }
    }
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "export",
    skip_all,
    fields(origin = "user", format = ?format),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn export_pages(
    window: WebviewWindow<Cef>,
    pages: Vec<EntityId>,
    format: ExportFormat,
    handle: AppHandle<Cef>,
) -> std::result::Result<(), Error> {
    let Some(directory) = rfd::AsyncFileDialog::new()
        .set_parent(&window)
        .pick_folder()
        .await
        .map(|directory| directory.path().to_owned())
    else {
        return Ok(());
    };
    let directory = validate_export_directory(&directory)?;
    export_pages_to_directory(&handle, pages, format, directory).await?;
    Ok(())
}

pub(crate) fn validate_export_directory(path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        bail!(
            "export directory must be an absolute path: {}",
            path.display()
        );
    }
    let path = path
        .canonicalize()
        .with_context(|| format!("failed to resolve export directory {}", path.display()))?;
    if !path.is_dir() {
        bail!("export path is not a directory: {}", path.display());
    }
    Ok(path)
}

pub(crate) async fn export_pages_to_directory(
    handle: &AppHandle<Cef>,
    pages: Vec<EntityId>,
    format: ExportFormat,
    directory: PathBuf,
) -> Result<Vec<PathBuf>> {
    let snapshot = {
        let current = handle.state::<CurrentProject>();
        let project = current.project.lock().await;
        let project = project.as_ref().context("no project is open")?;
        project.snapshot()
    };
    let desktop = handle.state::<Desktop>();
    let renderer = desktop.renderer();
    let rasterizer = desktop.rasterizer().await?;
    koharu_desktop::export_pages(
        renderer,
        rasterizer,
        snapshot,
        pages,
        format.into(),
        directory,
    )
    .await
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_thumbnail(
    page: EntityId,
    project: State<'_, CurrentProject>,
) -> std::result::Result<ThumbnailBytes, Error> {
    let snapshot = project
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    snapshot.page(page)?;
    let blob = snapshot
        .asset(page, &AssetRole::new("source")?)?
        .with_context(|| format!("page {page} has no source image"))?
        .blob;
    let bytes = snapshot.read_blob(blob).await?;
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
        let image = image::load_from_memory(&bytes).context("failed to decode source image")?;
        if image.width() == 0 || image.height() == 0 {
            return Err(anyhow::anyhow!("source image is empty"));
        }
        let image = image.thumbnail(THUMBNAIL_EDGE, THUMBNAIL_EDGE).to_rgba8();
        let encoder = webp::Encoder::from_rgba(image.as_raw(), image.width(), image.height());
        Ok(encoder.encode(80.0).to_vec())
    })
    .await
    .context("thumbnail worker stopped unexpectedly")??;
    Ok(ThumbnailBytes(bytes))
}

pub(crate) async fn rendered_preview(
    renderer: &Renderer,
    rasterizer: Arc<Rasterizer>,
    snapshot: &Snapshot,
    page: EntityId,
) -> Result<Vec<u8>> {
    koharu_desktop::rendered_preview(renderer, rasterizer, snapshot, page).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_export_path_must_be_an_absolute_directory() {
        assert!(validate_export_directory(Path::new("exports")).is_err());

        let directory =
            std::env::temp_dir().join(format!("koharu-export-path-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let file = directory.join("not-a-directory");
        std::fs::write(&file, []).unwrap();

        assert_eq!(
            validate_export_directory(&directory).unwrap(),
            directory.canonicalize().unwrap()
        );
        assert!(validate_export_directory(&file).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
