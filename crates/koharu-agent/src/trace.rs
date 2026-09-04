use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context as _, Result};
use serde::Serialize;
use serde_json::{Value, json};
use specta::Type;

use crate::RunId;

const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Type)]
pub struct TraceLocation {
    pub run: RunId,
    pub path: String,
}

pub fn trace_location(run: RunId) -> Result<TraceLocation> {
    TraceLocation::new(run, trace_path(run)?)
}

impl TraceLocation {
    pub fn explicit(run: RunId, path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.is_absolute() {
            anyhow::bail!("agent trace path must be absolute: {}", path.display());
        }
        let parent = path
            .parent()
            .context("the agent trace path has no parent directory")?
            .canonicalize()
            .with_context(|| {
                format!(
                    "failed to resolve agent trace directory {}",
                    path.parent().expect("trace parent was checked").display()
                )
            })?;
        if !parent.is_dir() {
            anyhow::bail!(
                "agent trace parent is not a directory: {}",
                parent.display()
            );
        }
        let file_name = path
            .file_name()
            .context("the agent trace path must name a file")?;
        Self::new(run, parent.join(file_name))
    }

    fn new(run: RunId, path: PathBuf) -> Result<Self> {
        let path = path
            .to_str()
            .context("the agent trace path is not valid UTF-8")?
            .to_owned();
        Ok(Self { run, path })
    }

    pub(crate) fn path(&self) -> &Path {
        Path::new(&self.path)
    }
}

pub(crate) struct Trace {
    run: RunId,
    sequence: u64,
    writer: BufWriter<File>,
}

impl Trace {
    pub(crate) fn create(location: &TraceLocation) -> Result<Self> {
        let run = location.run;
        let path = location.path();
        let directory = path
            .parent()
            .context("the agent trace path has no parent directory")?;
        std::fs::create_dir_all(directory)
            .with_context(|| format!("failed to create {}", directory.display()))?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        Ok(Self {
            run,
            sequence: 0,
            writer: BufWriter::new(file),
        })
    }

    pub(crate) fn record(&mut self, event: &str, data: Value) -> Result<()> {
        let timestamp_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before the Unix epoch")?
            .as_millis();
        let record = json!({
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run,
            "sequence": self.sequence,
            "timestamp_unix_ms": timestamp_unix_ms,
            "event": event,
            "data": data,
        });
        serde_json::to_writer(&mut self.writer, &record)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        self.sequence += 1;
        Ok(())
    }
}

fn trace_path(run: RunId) -> Result<PathBuf> {
    let config = koharu_config::path()?;
    let root = config
        .parent()
        .context("the Koharu configuration path has no parent directory")?;
    Ok(root
        .join("traces")
        .join("agent")
        .join(format!("{run}.jsonl")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_records_are_sequenced_json_lines() {
        let run = RunId::new();
        let directory = std::env::temp_dir().join(format!("koharu-agent-trace-{run}"));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("trace.jsonl");
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let mut trace = Trace {
            run,
            sequence: 0,
            writer: BufWriter::new(file),
        };
        trace.record("first", json!({ "value": 1 })).unwrap();
        trace.record("second", json!({ "value": 2 })).unwrap();
        drop(trace);

        let records = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        std::fs::remove_dir_all(directory).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["sequence"], 0);
        assert_eq!(records[1]["sequence"], 1);
        assert_eq!(records[0]["run_id"], run.to_string());
    }

    #[test]
    fn explicit_trace_locations_require_an_absolute_path_and_existing_parent() {
        let run = RunId::new();
        assert!(TraceLocation::explicit(run, "trace.jsonl").is_err());

        let directory = tempfile::tempdir().unwrap();
        let location = TraceLocation::explicit(run, directory.path().join("trace.jsonl")).unwrap();
        assert_eq!(Path::new(&location.path).parent(), Some(directory.path()));
    }
}
