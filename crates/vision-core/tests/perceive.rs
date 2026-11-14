//! The whole flow on a synthetic screenshot, with fake engines: what comes out, and that it is
//! valid protocol.

use image::{DynamicImage, Rgb, RgbImage};
use organon_protocol::{Location, Message};
use organon_vision_core::emit::{messages, Stored, LAYOUT_BLOCK, OBJECT, TEXT_LINE};
use organon_vision_core::objects::{Object, ObjectDetector};
use organon_vision_core::text::{TextLine, TextReader, Word};
use organon_vision_core::{perceive, Engines, Region};

/// A white page: a heading, a two-line paragraph (where the fake reader says), and a dark
/// line drawing on the right.
fn page() -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(800, 600, |x, y| {
        let drawing = (500..760).contains(&x) && (200..460).contains(&y);
        if drawing && (x % 5 == 0 || y % 7 == 0) {
            Rgb([30, 60, 120])
        } else {
            Rgb([250, 250, 250])
        }
    }))
}

struct FakeReader;

fn word(text: &str, x: u32, y: u32, w: u32, h: u32) -> Word {
    Word {
        text: text.to_owned(),
        region: Region::new(x, y, w, h),
    }
}

impl TextReader for FakeReader {
    fn tool(&self) -> String {
        "fake-ocr 1".to_owned()
    }
    fn read(&self, _img: &DynamicImage) -> organon_vision_core::error::Result<Vec<TextLine>> {
        let lines = [
            vec![word("Results", 40, 40, 200, 40)],
            vec![
                word("first", 40, 200, 80, 16),
                word("line", 130, 200, 60, 16),
            ],
            vec![word("second", 40, 220, 90, 16)],
            vec![word("x", 700, 580, 8, 8)], // lone character: dropped by a real engine wrapper
        ];
        Ok(lines.into_iter().filter_map(TextLine::from_words).collect())
    }
}

struct FakeDetector;

impl ObjectDetector for FakeDetector {
    fn tool(&self) -> String {
        "fake-det 1".to_owned()
    }
    fn model(&self) -> String {
        "toy.onnx".to_owned()
    }
    fn detect(&self, _img: &DynamicImage) -> organon_vision_core::error::Result<Vec<Object>> {
        Ok(vec![Object {
            label: "kite".to_owned(),
            score: 0.875,
            region: Region::new(500, 200, 260, 260),
        }])
    }
}

const STORED: Stored = Stored {
    uri: "page.png",
    title: "page.png",
    image_path: "/tmp/page.png",
    format: "png",
    captured_at: "2026-11-14T00:00:00Z",
    version: "0.1.0",
};

#[test]
fn a_page_becomes_lines_blocks_palette_and_objects() {
    let engines = Engines {
        text: Some(&FakeReader),
        objects: Some(&FakeDetector),
    };
    let p = perceive(&page(), engines).unwrap();
    assert_eq!(p.lines.len(), 4);
    assert_eq!(p.lines[1].text, "first line");
    assert_eq!(p.palette[0].hex, "#fafafa", "{:?}", p.palette);
    let text_blocks: Vec<_> = p.blocks.iter().filter(|b| !b.lines.is_empty()).collect();
    assert_eq!(text_blocks.len(), 3, "{:?}", p.blocks);
    assert_eq!(p.block_text(text_blocks[1]), "first line\nsecond");
    let figures = p.blocks.len() - text_blocks.len();
    assert_eq!(figures, 1, "{:?}", p.blocks);
    let md = p.summary_markdown("page.png");
    assert!(
        md.contains("text [40,200,150,36] 16 px: first line / second"),
        "{md}"
    );
    assert!(md.contains("objects: kite 0.88 [500,200,260,260]"), "{md}");
}

#[test]
fn messages_are_valid_protocol_with_regions_in_pixels() {
    let engines = Engines {
        text: Some(&FakeReader),
        objects: Some(&FakeDetector),
    };
    let p = perceive(&page(), engines).unwrap();
    let m = messages(&p, b"bytes", &STORED);
    let lines: String = m
        .iter()
        .map(|m| serde_json::to_string(m).unwrap() + "\n")
        .collect();
    let report = organon_protocol::check_stream(lines.as_bytes()).unwrap();
    assert!(report.is_ok(), "{report:?}");
    let obs: Vec<_> = m
        .iter()
        .filter_map(|m| match m {
            Message::Observation(o) => Some(o),
            _ => None,
        })
        .collect();
    let object = obs.iter().find(|o| o.kind.as_str() == OBJECT).unwrap();
    assert_eq!(
        object.at,
        Location::Page {
            page: 1,
            bbox: Some([500.0, 200.0, 260.0, 260.0])
        }
    );
    assert_eq!(object.produced_by.model.as_deref(), Some("toy.onnx"));
    // a text block points at its lines by observation id
    let line_ids: Vec<String> = obs
        .iter()
        .filter(|o| o.kind.as_str() == TEXT_LINE)
        .map(|o| o.id.to_string())
        .collect();
    let block = obs
        .iter()
        .find(|o| o.kind.as_str() == LAYOUT_BLOCK && o.value["text"] == "first line\nsecond")
        .unwrap();
    for id in block.value["lines"].as_array().unwrap() {
        assert!(line_ids.contains(&id.as_str().unwrap().to_owned()));
    }
}

#[test]
fn without_engines_nothing_is_claimed_about_text_or_objects() {
    let p = perceive(&page(), Engines::default()).unwrap();
    assert!(p.lines.is_empty() && p.objects.is_empty());
    let md = p.summary_markdown("page.png");
    assert!(md.contains("text: not read (no text engine)"), "{md}");
    assert!(md.contains("objects: not looked for"), "{md}");
    let m = messages(&p, b"bytes", &STORED);
    assert!(!m
        .iter()
        .any(|m| matches!(m, Message::Observation(o) if o.kind.as_str() == OBJECT)));
}

#[test]
fn perceiving_twice_gives_the_same_ids() {
    let run = || {
        let p = perceive(&page(), Engines::default()).unwrap();
        messages(&p, b"bytes", &STORED)
            .iter()
            .map(|m| serde_json::to_string(m).unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}
