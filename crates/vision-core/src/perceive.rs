//! The perception flow for one image: measures and palette always, text and objects when an
//! engine is given, layout from text and pixels.

use std::fmt::Write as _;

use image::DynamicImage;
use serde::Serialize;

use crate::error::Result;
use crate::layout::{blocks, Block, Role};
use crate::measure::{Features, Look, LookRules};
use crate::objects::{Object, ObjectDetector};
use crate::palette::{dominant, PaletteRules, Swatch};
use crate::text::{TextLine, TextReader};

/// The engines available for this run. `None` means "not available", never "nothing there".
#[derive(Default, Clone, Copy)]
pub struct Engines<'a> {
    pub text: Option<&'a dyn TextReader>,
    pub objects: Option<&'a dyn ObjectDetector>,
}

/// Everything perceived in one image.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Perception {
    pub width: u32,
    pub height: u32,
    pub features: Features,
    pub look: Look,
    pub palette: Vec<Swatch>,
    /// Text lines, top to bottom. Empty when no text engine ran (see `text_tool`).
    pub lines: Vec<TextLine>,
    /// Layout blocks in reading order; text blocks index into `lines`.
    pub blocks: Vec<Block>,
    pub objects: Vec<Object>,
    /// The text engine that ran, if any.
    pub text_tool: Option<String>,
    /// The object engine and model that ran, if any.
    pub object_tool: Option<(String, String)>,
}

/// Perceive one decoded image (transparent pixels are seen over white).
///
/// # Errors
/// An engine failed.
pub fn perceive(img: &DynamicImage, engines: Engines) -> Result<Perception> {
    let img = &*crate::sense::flatten(img);
    let features = Features::of(img);
    let mut lines = match engines.text {
        Some(reader) => reader.read(img)?,
        None => Vec::new(),
    };
    lines.sort_by_key(|l| (l.region.y, l.region.x));
    let objects = match engines.objects {
        Some(d) => d.detect(img)?,
        None => Vec::new(),
    };
    Ok(Perception {
        width: img.width(),
        height: img.height(),
        look: features.look(LookRules::default()),
        features,
        palette: dominant(img, PaletteRules::default()),
        blocks: blocks(img, &lines),
        lines,
        objects,
        text_tool: engines.text.map(TextReader::tool),
        object_tool: engines.objects.map(|d| (d.tool(), d.model())),
    })
}

impl Perception {
    /// The text of a block, its lines joined by newlines.
    #[must_use]
    pub fn block_text(&self, block: &Block) -> String {
        block
            .lines
            .iter()
            .map(|&i| self.lines[i].text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// A compact Markdown view for a reader: what to look at, in reading order.
    #[must_use]
    pub fn summary_markdown(&self, title: &str) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "# {title} · {}×{} · {} · ink {:.2}",
            self.width,
            self.height,
            look_name(self.look),
            self.features.ink
        );
        let colours: Vec<String> = self
            .palette
            .iter()
            .map(|c| format!("{} {:.0}%", c.hex, c.share * 100.0))
            .collect();
        let _ = writeln!(s, "palette: {}", colours.join(" · "));
        match &self.text_tool {
            Some(t) => {
                let _ = writeln!(s, "text: {} lines ({t})", self.lines.len());
            }
            None => s.push_str("text: not read (no text engine)\n"),
        }
        for (i, b) in self.blocks.iter().enumerate() {
            let [x, y, w, h]: [u32; 4] = b.region.into();
            let _ = match b.role {
                Role::Text => writeln!(
                    s,
                    "{}. text [{x},{y},{w},{h}] {} px: {}",
                    i + 1,
                    b.line_height.unwrap_or_default(),
                    preview(&self.block_text(b).replace('\n', " / "), 160)
                ),
                Role::Figure => writeln!(
                    s,
                    "{}. figure [{x},{y},{w},{h}] {}",
                    i + 1,
                    b.look.map_or("", look_name)
                ),
            };
        }
        match &self.object_tool {
            Some((_, model)) if self.objects.is_empty() => {
                let _ = writeln!(s, "objects: none found ({model})");
            }
            Some(_) => {
                let found: Vec<String> = self
                    .objects
                    .iter()
                    .map(|o| {
                        let [x, y, w, h]: [u32; 4] = o.region.into();
                        format!("{} {:.2} [{x},{y},{w},{h}]", o.label, o.score)
                    })
                    .collect();
                let _ = writeln!(s, "objects: {}", found.join(" · "));
            }
            None => s.push_str("objects: not looked for (no object model)\n"),
        }
        s
    }
}

fn look_name(l: Look) -> &'static str {
    match l {
        Look::Dark => "dark",
        Look::Graphic => "graphic",
        Look::Scene => "scene",
    }
}

fn preview(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let cut: String = text.chars().take(max_chars).collect();
    format!("{cut}…")
}
