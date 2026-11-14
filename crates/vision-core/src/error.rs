//! Typed errors of image perception.

/// What can go wrong while perceiving an image.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The bytes are not an image we can decode.
    #[error("image: {0}")]
    Decode(#[from] image::ImageError),
    /// The network (or a fake of it) failed for this URL.
    #[error("GET {url}: {reason}")]
    Fetch { url: String, reason: String },
    /// An engine (OCR, object detector) failed on this image.
    #[error("{engine}: {reason}")]
    Engine {
        engine: &'static str,
        reason: String,
    },
    /// A model file could not be loaded.
    #[error("model {what}: {reason}")]
    Model { what: String, reason: String },
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;
