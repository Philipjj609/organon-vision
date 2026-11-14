//! Engines that run neural models in pure Rust (feature `models`): [`Ocrs`] reads text with the
//! ocrs models, [`Yolo`] finds objects with a YOLOv8/YOLO11 detection model on RTen. Both take
//! model *bytes*: where models live and how they are downloaded is the binary's business.

use image::{DynamicImage, Rgb, RgbImage};
use ocrs::{ImageSource, OcrEngine, OcrEngineParams, TextItem};
use rten::Model;
use rten_tensor::prelude::*;
use rten_tensor::NdTensor;

use crate::error::{Error, Result};
use crate::geom::Region;
use crate::objects::{decode_yolo, Letterbox, Object, ObjectDetector};
use crate::text::{tidy, TextLine, TextReader, Word};

/// The ocrs OCR engine: word detection, line grouping, line recognition. Latin script only.
pub struct Ocrs {
    engine: OcrEngine,
}

impl Ocrs {
    /// An engine from the detection and recognition models (`.rten` or `.onnx` bytes).
    ///
    /// # Errors
    /// A model does not load.
    pub fn from_bytes(detection: Vec<u8>, recognition: Vec<u8>) -> Result<Self> {
        let load = |what: &str, bytes| {
            Model::load(bytes).map_err(|e| Error::Model {
                what: what.to_owned(),
                reason: e.to_string(),
            })
        };
        let engine = OcrEngine::new(OcrEngineParams {
            detection_model: Some(load("text-detection", detection)?),
            recognition_model: Some(load("text-recognition", recognition)?),
            ..OcrEngineParams::default()
        })
        .map_err(|e| engine_error("ocrs", e))?;
        Ok(Self { engine })
    }
}

fn engine_error(engine: &'static str, e: impl std::fmt::Display) -> Error {
    Error::Engine {
        engine,
        reason: e.to_string(),
    }
}

fn region(r: &rten_imageproc::Rect) -> Region {
    Region::from_edges(
        i64::from(r.left()),
        i64::from(r.top()),
        i64::from(r.right()),
        i64::from(r.bottom()),
    )
}

impl TextReader for Ocrs {
    fn tool(&self) -> String {
        "ocrs 0.14".to_owned()
    }

    fn read(&self, img: &DynamicImage) -> Result<Vec<TextLine>> {
        let rgb = img.to_rgb8();
        let source = ImageSource::from_bytes(rgb.as_raw(), rgb.dimensions())
            .map_err(|e| engine_error("ocrs", e))?;
        let input = self
            .engine
            .prepare_input(source)
            .map_err(|e| engine_error("ocrs", e))?;
        let words = self
            .engine
            .detect_words(&input)
            .map_err(|e| engine_error("ocrs", e))?;
        let line_rects = self.engine.find_text_lines(&input, &words);
        let lines = self
            .engine
            .recognize_text(&input, &line_rects)
            .map_err(|e| engine_error("ocrs", e))?;
        let lines = lines.iter().flatten().filter_map(|line| {
            let words = line
                .words()
                .map(|w| Word {
                    text: w.to_string(),
                    region: region(&w.bounding_rect()),
                })
                .collect();
            TextLine::from_words(words)
        });
        Ok(tidy(lines.collect()))
    }
}

/// A YOLOv8/YOLO11-style detector (output `[1, 4 + classes, anchors]`) on RTen.
pub struct Yolo {
    model: Model,
    labels: Vec<String>,
    name: String,
    side: u32,
    min_score: f32,
}

/// Input side of the standard exports.
const YOLO_SIDE: u32 = 640;
/// Letterbox padding grey, as in Ultralytics.
const PAD: u8 = 114;

impl Yolo {
    /// A detector from model bytes, its class names (empty: the 80 COCO classes) and a name for
    /// `produced_by.model` (the file name is fine).
    ///
    /// # Errors
    /// The model does not load.
    pub fn from_bytes(model: Vec<u8>, labels: Vec<String>, name: &str) -> Result<Self> {
        let model = Model::load(model).map_err(|e| Error::Model {
            what: name.to_owned(),
            reason: e.to_string(),
        })?;
        let labels = if labels.is_empty() {
            COCO.iter().map(|&s| s.to_owned()).collect()
        } else {
            labels
        };
        Ok(Self {
            model,
            labels,
            name: name.to_owned(),
            side: YOLO_SIDE,
            min_score: 0.35,
        })
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn input(&self, img: &DynamicImage) -> (NdTensor<f32, 4>, Letterbox) {
        let lb = Letterbox::fit(img.width(), img.height(), self.side);
        let w = ((img.width() as f32 * lb.scale).round() as u32).clamp(1, self.side);
        let h = ((img.height() as f32 * lb.scale).round() as u32).clamp(1, self.side);
        let resized = img
            .resize_exact(w, h, image::imageops::FilterType::Triangle)
            .to_rgb8();
        let mut canvas = RgbImage::from_pixel(self.side, self.side, Rgb([PAD; 3]));
        image::imageops::overlay(
            &mut canvas,
            &resized,
            i64::from(lb.pad_x as u32),
            i64::from(lb.pad_y as u32),
        );
        let s = self.side as usize;
        let mut data = vec![0.0_f32; 3 * s * s];
        for (x, y, p) in canvas.enumerate_pixels() {
            for c in 0..3 {
                data[c * s * s + y as usize * s + x as usize] = f32::from(p[c]) / 255.0;
            }
        }
        (NdTensor::from_data([1, 3, s, s], data), lb)
    }
}

impl ObjectDetector for Yolo {
    fn tool(&self) -> String {
        "rten 0.28".to_owned()
    }

    fn model(&self) -> String {
        self.name.clone()
    }

    fn detect(&self, img: &DynamicImage) -> Result<Vec<Object>> {
        let (input, lb) = self.input(img);
        let out = self
            .model
            .run_one(input.view().into(), None)
            .map_err(|e| engine_error("rten", e))?;
        let out: NdTensor<f32, 3> = out.try_into().map_err(|e| engine_error("rten", e))?;
        let [_, rows, anchors] = out.shape();
        let classes = rows.saturating_sub(4);
        let labels: Vec<String> = if classes == self.labels.len() {
            self.labels.clone()
        } else {
            (0..classes).map(|c| format!("class_{c}")).collect()
        };
        let data = out.to_vec();
        Ok(decode_yolo(&data, &labels, anchors, lb, self.min_score))
    }
}

/// The 80 COCO class names, in the order YOLO exports use.
pub const COCO: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];
