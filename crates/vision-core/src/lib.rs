//! Image perception for organon: look at a still image (a screenshot, a diagram, a photo) and
//! describe it as located organon-protocol observations.
//!
//! - [`measure`]: whole-image measures (brightness, ink, difference hash) and coarse look.
//!   Shared with organon-video, which measures its keyframes with it.
//! - [`palette`]: dominant colours (k-means in CIE Lab).
//! - [`text`]: text lines and words, behind the [`text::TextReader`] trait.
//! - [`objects`]: objects behind the [`objects::ObjectDetector`] trait, plus YOLO decoding.
//! - [`layout`]: text blocks and figure blocks, from geometry.
//! - [`engines`] (feature `models`): ocrs for text and RTen for YOLO objects, pure Rust.
//! - [`sense`]: image bytes from a file or a URL, behind the [`sense::Fetch`] trait.
//! - [`perceive`]: the flow; [`emit`]: protocol messages.
//!
//! Perception does not interpret: a block is "text" or "figure", never "title" or "chart"; an
//! object label is the model's, with its score. No file access, network or printing here.

#![forbid(unsafe_code)]

pub mod emit;
#[cfg(feature = "models")]
pub mod engines;
pub mod error;
pub mod geom;
pub mod layout;
pub mod measure;
pub mod objects;
pub mod palette;
pub mod perceive;
pub mod sense;
pub mod text;

pub use error::Error;
pub use geom::Region;
pub use perceive::{perceive, Engines, Perception};
