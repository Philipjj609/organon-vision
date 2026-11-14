//! A perception as organon-protocol messages: one source, then located observations. Regions
//! are `Location::Page { page: 1, bbox: [x, y, w, h] }` in pixels of the original image.

use std::collections::BTreeMap;

use organon_protocol::{
    sha256_hex, Location, Message, Observation, ObservationKind, ProducedBy, Source, SourceId,
};
use serde_json::json;

use crate::geom::Region;
use crate::layout::Role;
use crate::perceive::Perception;

/// Organ name in `produced_by`.
pub const ORGAN: &str = "organon-vision";
/// Whole image: size and measures (`Location::Whole`).
pub const IMAGE: &str = "image";
/// Dominant colours (`Location::Whole`).
pub const PALETTE: &str = "palette";
/// One line of text, with its words (`Location::Page` + bbox).
pub const TEXT_LINE: &str = "text_line";
/// A layout block: text (with its line ids) or figure (`Location::Page` + bbox).
pub const LAYOUT_BLOCK: &str = "layout_block";
/// A detected object (`Location::Page` + bbox).
pub const OBJECT: &str = "object";

/// Where and when an image was stored, and by which version.
#[derive(Debug, Clone)]
pub struct Stored<'a> {
    /// Where it came from: a URL or the path given.
    pub uri: &'a str,
    pub title: &'a str,
    /// Path of the stored copy, recorded as `files["image"]`.
    pub image_path: &'a str,
    /// `png`, `jpeg`, …
    pub format: &'a str,
    /// RFC 3339.
    pub captured_at: &'a str,
    /// This organ's version.
    pub version: &'a str,
}

/// The source id of an image: the hash of its bytes, so the same picture is the same source
/// wherever it came from.
#[must_use]
pub fn source_id(bytes: &[u8]) -> SourceId {
    SourceId::of(bytes)
}

/// Messages for one image: the source, then observations (image, palette, text lines, layout
/// blocks, objects).
#[must_use]
pub fn messages(p: &Perception, bytes: &[u8], stored: &Stored) -> Vec<Message> {
    let by = |tool: Option<String>, model: Option<String>| ProducedBy {
        organ: ORGAN.to_owned(),
        version: stored.version.to_owned(),
        tool,
        model,
    };
    let id = source_id(bytes);
    let mut out = vec![Message::Source(Source {
        id: id.clone(),
        kind: "image".to_owned(),
        uri: stored.uri.to_owned(),
        title: stored.title.to_owned(),
        captured_at: stored.captured_at.to_owned(),
        content_hash: sha256_hex(bytes),
        files: BTreeMap::from([("image".to_owned(), stored.image_path.to_owned())]),
        meta: BTreeMap::from([
            ("width".to_owned(), json!(p.width)),
            ("height".to_owned(), json!(p.height)),
            ("format".to_owned(), json!(stored.format)),
        ]),
        produced_by: Some(by(None, None)),
    })];
    let observe = |kind: &str, at: Location, value: serde_json::Value, by: ProducedBy| {
        Observation::new(id.clone(), ObservationKind::new(kind), at, value, by)
    };
    let page = |r: Region| Location::Page {
        page: 1,
        bbox: Some(r.to_bbox()),
    };
    let f = &p.features;
    out.push(Message::Observation(observe(
        IMAGE,
        Location::Whole {},
        json!({
            "width": p.width,
            "height": p.height,
            "look": p.look,
            "mean": round3(f.mean),
            "white": round3(f.white),
            "ink": round3(f.ink),
            "dhash": format!("{:016x}", f.dhash),
        }),
        by(None, None),
    )));
    out.push(Message::Observation(observe(
        PALETTE,
        Location::Whole {},
        json!({ "colors": p.palette, "method": "k-means in CIE Lab, k-means++ seeding" }),
        by(Some("kmeans_colors 0.7".to_owned()), None),
    )));
    let lines: Vec<Observation> = p
        .lines
        .iter()
        .map(|l| {
            let words: Vec<_> = l
                .words
                .iter()
                .map(|w| json!({ "text": w.text, "bbox": w.region }))
                .collect();
            observe(
                TEXT_LINE,
                page(l.region),
                json!({ "text": l.text, "words": words }),
                by(p.text_tool.clone(), None),
            )
        })
        .collect();
    for b in &p.blocks {
        let value = match b.role {
            Role::Text => json!({
                "role": b.role,
                "text": p.block_text(b),
                "line_height": b.line_height,
                "lines": b.lines.iter().map(|&i| lines[i].id.to_string()).collect::<Vec<_>>(),
            }),
            Role::Figure => json!({ "role": b.role, "look": b.look }),
        };
        let observation = observe(LAYOUT_BLOCK, page(b.region), value, by(None, None));
        out.push(Message::Observation(observation));
    }
    out.extend(lines.into_iter().map(Message::Observation));
    if let Some((tool, model)) = &p.object_tool {
        out.extend(p.objects.iter().map(|o| {
            Message::Observation(observe(
                OBJECT,
                page(o.region),
                json!({ "label": o.label, "score": o.score }),
                by(Some(tool.clone()), Some(model.clone())),
            ))
        }));
    }
    out
}

fn round3(x: f32) -> f64 {
    (f64::from(x) * 1000.0).round() / 1000.0
}
