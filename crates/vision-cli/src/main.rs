//! `organon-vision`: look at still images (files or URLs) and write organon-protocol messages,
//! ready for `mneme ingest`, plus a short Markdown view for a reader.
//!
//! Work folder: `$ORGANON_HOME/vision` (`ORGANON_HOME` = `~/.organon`). Stored copies go to
//! `sources/<source id>.<ext>` (observation regions point into them), runs to
//! `runs/<time>.jsonl`, models to `models/`. Third-party images stay there, never in a repo.

#![forbid(unsafe_code)]

mod net;

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::Context as _;
use clap::{Parser, Subcommand};
use organon_protocol::Message;
use organon_vision_core::emit::{messages, source_id, Stored};
use organon_vision_core::engines::{Ocrs, Yolo};
use organon_vision_core::sense::{Fetch, Sensed};
use organon_vision_core::{perceive, Engines};

/// ocrs models in ONNX form (RTen loads ONNX directly). The `.rten` copies its older CLI used
/// answer 403 since 2026 (measured 2026-11-14); the ocrs README warns these URLs may move, so the
/// files are copied into the work folder once.
const OCRS_MODELS: [(&str, &str); 2] = [
    (
        "text-detection.onnx",
        "https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.onnx",
    ),
    (
        "text-recognition.onnx",
        "https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.onnx",
    ),
];
/// Object model file names looked for in the models folder.
const OBJECT_MODEL: &str = "objects.onnx";
const OBJECT_LABELS: &str = "objects.labels";

#[derive(Parser)]
#[command(
    name = "organon-vision",
    version,
    about = "See still images: text, layout blocks, palette and objects as located observations"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Which models and settings are present
    Doctor,
    /// Download the ocrs text models (about 12 MB); optionally fetch an object model
    Models {
        /// A YOLOv8/YOLO11 ONNX detection model (URL or file), stored as models/objects.onnx
        #[arg(long)]
        objects: Option<String>,
    },
    /// Look at images: files or URLs
    See {
        inputs: Vec<String>,
        /// Do not read text (faster; no OCR)
        #[arg(long)]
        no_text: bool,
        /// Do not look for objects
        #[arg(long)]
        no_objects: bool,
        /// Write messages here (default: $ORGANON_HOME/vision/runs/<time>.jsonl)
        #[arg(long)]
        out: Option<PathBuf>,
        /// Write messages to stdout instead of the summary (logs stay on stderr)
        #[arg(long)]
        json: bool,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> anyhow::Result<ExitCode> {
    let work = organon_home().join("vision");
    match cli.cmd {
        Cmd::Doctor => {
            doctor(&work);
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Models { objects } => {
            models(&work, objects.as_deref(), &net::Http::new())?;
            Ok(ExitCode::SUCCESS)
        }
        Cmd::See {
            inputs,
            no_text,
            no_objects,
            out,
            json,
        } => see(
            &work,
            &inputs,
            Wanted {
                text: !no_text,
                objects: !no_objects,
            },
            out.as_deref(),
            json,
        ),
    }
}

fn organon_home() -> PathBuf {
    std::env::var_os("ORGANON_HOME").map_or_else(
        || {
            let home = std::env::var_os("USERPROFILE")
                .or_else(|| std::env::var_os("HOME"))
                .unwrap_or_default();
            PathBuf::from(home).join(".organon")
        },
        PathBuf::from,
    )
}

fn doctor(work: &Path) {
    let models = work.join("models");
    println!("work folder: {}", work.display());
    let have_text = OCRS_MODELS.iter().all(|(f, _)| models.join(f).is_file());
    println!(
        "text (ocrs): {}",
        if have_text {
            "ready"
        } else {
            "models MISSING (run `organon-vision models`)"
        }
    );
    let object_model = models.join(OBJECT_MODEL);
    println!(
        "objects: {}",
        if object_model.is_file() {
            format!("ready ({})", object_model.display())
        } else {
            format!(
                "no model (put a YOLOv8/YOLO11 ONNX export at {}, or `organon-vision models --objects <url>`)",
                object_model.display()
            )
        }
    );
    println!("palette, layout, measures: always (no model)");
    println!("network: ocrs-models.s3-accelerate.amazonaws.com (models), and the URLs you give");
}

fn models(work: &Path, objects: Option<&str>, fetch: &dyn Fetch) -> anyhow::Result<()> {
    let dir = work.join("models");
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    for (file, url) in OCRS_MODELS {
        let path = dir.join(file);
        if path.is_file() {
            eprintln!("{file}: present");
            continue;
        }
        let bytes = fetch
            .get(url)
            .map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;
        std::fs::write(&path, &bytes).with_context(|| format!("writing {}", path.display()))?;
        eprintln!("{file}: {} bytes", bytes.len());
    }
    if let Some(src) = objects {
        let bytes = if src.starts_with("http") {
            fetch
                .get(src)
                .map_err(|e| anyhow::anyhow!("GET {src}: {e}"))?
        } else {
            std::fs::read(src).with_context(|| format!("reading {src}"))?
        };
        Yolo::from_bytes(bytes.clone(), Vec::new(), OBJECT_MODEL)
            .context("the object model does not load")?;
        std::fs::write(dir.join(OBJECT_MODEL), &bytes)?;
        eprintln!("{OBJECT_MODEL}: {} bytes", bytes.len());
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Wanted {
    text: bool,
    objects: bool,
}

/// Load the engines that are wanted and present; say on stderr which are not.
fn engines(work: &Path, wanted: Wanted) -> anyhow::Result<(Option<Ocrs>, Option<Yolo>)> {
    let dir = work.join("models");
    let text = if wanted.text {
        let paths: Vec<PathBuf> = OCRS_MODELS.iter().map(|(f, _)| dir.join(f)).collect();
        if paths.iter().all(|p| p.is_file()) {
            Some(Ocrs::from_bytes(
                std::fs::read(&paths[0])?,
                std::fs::read(&paths[1])?,
            )?)
        } else {
            eprintln!("text: no ocrs models (run `organon-vision models`); text is not read");
            None
        }
    } else {
        None
    };
    let model = dir.join(OBJECT_MODEL);
    let objects = if wanted.objects && model.is_file() {
        let labels = std::fs::read_to_string(dir.join(OBJECT_LABELS))
            .map(|t| t.lines().map(str::trim).map(str::to_owned).collect())
            .unwrap_or_default();
        Some(Yolo::from_bytes(
            std::fs::read(&model)?,
            labels,
            OBJECT_MODEL,
        )?)
    } else {
        None
    };
    Ok((text, objects))
}

fn see(
    work: &Path,
    inputs: &[String],
    wanted: Wanted,
    out: Option<&Path>,
    json: bool,
) -> anyhow::Result<ExitCode> {
    anyhow::ensure!(
        !inputs.is_empty(),
        "nothing to see: give image files or URLs"
    );
    let (ocrs, yolo) = engines(work, wanted)?;
    let engines = Engines {
        text: ocrs
            .as_ref()
            .map(|e| e as &dyn organon_vision_core::text::TextReader),
        objects: yolo
            .as_ref()
            .map(|e| e as &dyn organon_vision_core::objects::ObjectDetector),
    };
    let http = net::Http::new();
    let sources = work.join("sources");
    std::fs::create_dir_all(&sources)?;
    let now = now_rfc3339();
    let version = env!("CARGO_PKG_VERSION");
    let mut all: Vec<Message> = Vec::new();
    let mut failures = 0;
    for (i, input) in inputs.iter().enumerate() {
        let started = std::time::Instant::now();
        let done = sense(input, &http).and_then(|s| {
            let img = s.decode()?;
            let p = perceive(&img, engines)?;
            let stored_path = sources.join(format!("{}.{}", source_id(&s.bytes), s.extension()));
            if !stored_path.is_file() {
                std::fs::write(&stored_path, &s.bytes)?;
            }
            let path = stored_path.to_string_lossy();
            let stored = Stored {
                uri: &s.uri,
                title: &s.title,
                image_path: &path,
                format: s.extension(),
                captured_at: &now,
                version,
            };
            Ok((
                p.summary_markdown(&s.title),
                messages(&p, &s.bytes, &stored),
            ))
        });
        match done {
            Ok((summary, msgs)) => {
                eprintln!(
                    "[{}/{}] {input} ok ({:.1} s, {} messages)",
                    i + 1,
                    inputs.len(),
                    started.elapsed().as_secs_f32(),
                    msgs.len()
                );
                if !json {
                    println!("{summary}");
                }
                all.extend(msgs);
            }
            Err(err) => {
                failures += 1;
                eprintln!("[{}/{}] {input} FAILED: {err:#}", i + 1, inputs.len());
            }
        }
    }
    write_messages(work, &all, out, json, &now)?;
    Ok(if failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    })
}

fn sense(input: &str, fetch: &dyn Fetch) -> anyhow::Result<Sensed> {
    if input.starts_with("http://") || input.starts_with("https://") {
        Ok(Sensed::from_url(input, fetch)?)
    } else {
        let bytes = std::fs::read(input).with_context(|| format!("reading {input}"))?;
        Ok(Sensed::from_bytes(input, bytes)?)
    }
}

fn write_messages(
    work: &Path,
    all: &[Message],
    out: Option<&Path>,
    json: bool,
    now: &str,
) -> anyhow::Result<()> {
    let mut text = String::new();
    for m in all {
        text.push_str(&serde_json::to_string(m)?);
        text.push('\n');
    }
    if json {
        std::io::stdout().write_all(text.as_bytes())?;
        return Ok(());
    }
    let path = match out {
        Some(p) => p.to_path_buf(),
        None => {
            let runs = work.join("runs");
            std::fs::create_dir_all(&runs)?;
            runs.join(format!("{}.jsonl", now.replace(':', "-")))
        }
    };
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
    eprintln!("{} messages → {}", all.len(), path.display());
    Ok(())
}

/// Current UTC time, RFC 3339, without a date crate: civil-from-days (Howard Hinnant).
fn now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let days = i64::try_from(secs / 86_400).unwrap_or_default();
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct Fake(RefCell<Vec<String>>);

    impl Fetch for Fake {
        fn get(&self, url: &str) -> Result<Vec<u8>, String> {
            self.0.borrow_mut().push(url.to_owned());
            Ok(b"model".to_vec())
        }
    }

    /// Regression: the `.rten` model URLs answer 403; the ONNX ones are what exists.
    #[test]
    fn models_fetches_the_onnx_text_models_once() {
        let work = std::env::temp_dir().join(format!("organon-vision-test-{}", std::process::id()));
        let fake = Fake(RefCell::new(Vec::new()));
        models(&work, None, &fake).unwrap();
        models(&work, None, &fake).unwrap();
        let urls = fake.0.borrow();
        assert_eq!(urls.len(), 2, "second run reuses the files: {urls:?}");
        assert!(urls.iter().all(|u| u.ends_with(".onnx")), "{urls:?}");
        assert!(work.join("models/text-detection.onnx").is_file());
        std::fs::remove_dir_all(&work).unwrap();
    }

    #[test]
    fn rfc3339_has_the_right_shape() {
        let t = now_rfc3339();
        assert_eq!(t.len(), 20, "{t}");
        assert!(t.starts_with("20") && t.ends_with('Z'), "{t}");
    }
}
