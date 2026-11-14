//! Objects in an image: a label, a score and a region. Detection is an engine's job; this module
//! holds what any YOLO-family detector needs around the model: letterboxing, decoding the output
//! tensor and non-maximum suppression, as pure functions.

use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::geom::Region;

/// A detected object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Object {
    pub label: String,
    /// Detector confidence 0–1, three decimals.
    pub score: f32,
    pub region: Region,
}

/// Finds objects. Implemented by model engines; tests use a fake.
pub trait ObjectDetector {
    /// Engine and model, recorded as `produced_by.tool` / `model`.
    fn tool(&self) -> String;
    fn model(&self) -> String;

    /// Objects, best first.
    ///
    /// # Errors
    /// The engine failed on this image.
    fn detect(&self, img: &DynamicImage) -> Result<Vec<Object>>;
}

/// How an image was fitted into the model's square input: scaled by `scale`, then padded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    /// Original image size.
    pub width: u32,
    pub height: u32,
}

impl Letterbox {
    /// Fit `width × height` into a `side × side` square, centred.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn fit(width: u32, height: u32, side: u32) -> Self {
        let scale = (side as f32 / width as f32).min(side as f32 / height as f32);
        Self {
            scale,
            pad_x: (side as f32 - width as f32 * scale) / 2.0,
            pad_y: (side as f32 - height as f32 * scale) / 2.0,
            width,
            height,
        }
    }

    /// A model-space box (centre x, centre y, width, height) in original pixels, clamped.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_region(&self, cx: f32, cy: f32, w: f32, h: f32) -> Region {
        let x0 = ((cx - w / 2.0 - self.pad_x) / self.scale).round() as i64;
        let y0 = ((cy - h / 2.0 - self.pad_y) / self.scale).round() as i64;
        let x1 = ((cx + w / 2.0 - self.pad_x) / self.scale).round() as i64;
        let y1 = ((cy + h / 2.0 - self.pad_y) / self.scale).round() as i64;
        Region::from_edges(
            x0,
            y0,
            x1.min(i64::from(self.width)),
            y1.min(i64::from(self.height)),
        )
    }
}

/// Decode a YOLOv8/YOLO11 output `[4 + classes, anchors]` (row-major, batch dropped): each
/// column is `cx, cy, w, h` then one score per class. Keeps the best class per anchor at or
/// above `min_score`, then suppresses overlaps per label.
#[must_use]
pub fn decode_yolo(
    out: &[f32],
    labels: &[String],
    anchors: usize,
    letterbox: Letterbox,
    min_score: f32,
) -> Vec<Object> {
    let classes = labels.len();
    if out.len() < (4 + classes) * anchors {
        return Vec::new();
    }
    let at = |row: usize, col: usize| out[row * anchors + col];
    let found = (0..anchors).filter_map(|a| {
        let (class, score) = (0..classes)
            .map(|c| (c, at(4 + c, a)))
            .max_by(|x, y| x.1.total_cmp(&y.1))?;
        (score >= min_score).then(|| Object {
            label: labels[class].clone(),
            score: round3(score),
            region: letterbox.to_region(at(0, a), at(1, a), at(2, a), at(3, a)),
        })
    });
    suppress(found.collect(), 0.45)
}

/// Non-maximum suppression per label: keep the best box, drop same-label boxes overlapping it
/// by more than `max_iou`. Output is best first; ties break by position, so it is deterministic.
#[must_use]
pub fn suppress(mut objects: Vec<Object>, max_iou: f32) -> Vec<Object> {
    objects.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| (a.region.y, a.region.x).cmp(&(b.region.y, b.region.x)))
    });
    let mut kept: Vec<Object> = Vec::new();
    for o in objects {
        if o.region.area() == 0 {
            continue;
        }
        if !kept
            .iter()
            .any(|k| k.label == o.label && k.region.iou(&o.region) > max_iou)
        {
            kept.push(o);
        }
    }
    kept
}

fn round3(x: f32) -> f32 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_maps_model_boxes_back_to_pixels() {
        // 1280×640 into 640: scale 0.5, 160 px of padding above and below
        let lb = Letterbox::fit(1280, 640, 640);
        assert_eq!((lb.scale, lb.pad_x, lb.pad_y), (0.5, 0.0, 160.0));
        assert_eq!(
            lb.to_region(320.0, 320.0, 100.0, 50.0),
            Region::new(540, 270, 200, 100)
        );
    }

    #[test]
    fn decode_keeps_best_class_and_suppresses_overlaps() {
        let labels = vec!["cat".to_owned(), "dog".to_owned()];
        // three anchors: two overlapping cats, one weak dog
        #[rustfmt::skip]
        let out = [
            100.0, 104.0, 400.0, // cx
            100.0, 100.0, 400.0, // cy
             50.0,  50.0,  40.0, // w
             50.0,  50.0,  40.0, // h
              0.9,   0.8,   0.1, // cat
              0.1,   0.2,  0.15, // dog
        ];
        let lb = Letterbox::fit(640, 640, 640);
        let objs = decode_yolo(&out, &labels, 3, lb, 0.25);
        assert_eq!(objs.len(), 1, "{objs:?}");
        assert_eq!(objs[0].label, "cat");
        assert_eq!(objs[0].score, 0.9);
        assert_eq!(objs[0].region, Region::new(75, 75, 50, 50));
    }

    #[test]
    fn a_short_tensor_decodes_to_nothing_instead_of_panicking() {
        let lb = Letterbox::fit(10, 10, 640);
        assert!(decode_yolo(&[0.0; 5], &["a".to_owned()], 3, lb, 0.1).is_empty());
    }
}
