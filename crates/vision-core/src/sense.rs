//! Sense: get an image's bytes and check that they are an image, behind the [`Fetch`] trait the
//! binary implements (tests use a fake), so no network code lives here.

use image::{DynamicImage, ImageFormat};

use crate::error::{Error, Result};

/// GET a URL. Implemented by the binary with real HTTP, and by a fake in tests.
pub trait Fetch {
    /// The body of a successful response.
    ///
    /// # Errors
    /// Any network or HTTP failure, as text.
    fn get(&self, url: &str) -> std::result::Result<Vec<u8>, String>;
}

/// An image as sensed: its bytes, where it came from, its format and a title.
#[derive(Debug, Clone)]
pub struct Sensed {
    pub uri: String,
    pub title: String,
    pub bytes: Vec<u8>,
    pub format: ImageFormat,
}

impl Sensed {
    /// Bytes from anywhere (a file the binary read, a download). The title is the last path
    /// segment of `uri` without a query.
    ///
    /// # Errors
    /// The bytes are not an image format we read.
    pub fn from_bytes(uri: &str, bytes: Vec<u8>) -> Result<Self> {
        let format = image::guess_format(&bytes)?;
        let title = uri
            .split(['?', '#'])
            .next()
            .unwrap_or(uri)
            .trim_end_matches(['/', '\\'])
            .rsplit(['/', '\\'])
            .next()
            .filter(|t| !t.is_empty())
            .unwrap_or(uri)
            .to_owned();
        Ok(Self {
            uri: uri.to_owned(),
            title,
            bytes,
            format,
        })
    }

    /// Download an image.
    ///
    /// # Errors
    /// The download failed or the body is not an image.
    pub fn from_url(url: &str, fetch: &dyn Fetch) -> Result<Self> {
        let bytes = fetch.get(url).map_err(|reason| Error::Fetch {
            url: url.to_owned(),
            reason,
        })?;
        Self::from_bytes(url, bytes)
    }

    /// Decode the pixels.
    ///
    /// # Errors
    /// The image is corrupt.
    pub fn decode(&self) -> Result<DynamicImage> {
        Ok(image::load_from_memory_with_format(
            &self.bytes,
            self.format,
        )?)
    }

    /// File extension for the stored copy (`png`, `jpg`, …).
    #[must_use]
    pub fn extension(&self) -> &'static str {
        self.format
            .extensions_str()
            .first()
            .copied()
            .unwrap_or("img")
    }
}

/// An image as a viewer sees it: transparent pixels composited over white (how a diagram shows
/// on a light page). Images without alpha are borrowed unchanged.
///
/// Regression (real run on two diagrams from GitHub): transparent backgrounds decoded as black,
/// so the palette said 85 % black for a diagram that looks white.
#[must_use]
pub fn flatten(img: &DynamicImage) -> std::borrow::Cow<'_, DynamicImage> {
    if !img.color().has_alpha() {
        return std::borrow::Cow::Borrowed(img);
    }
    let rgba = img.to_rgba8();
    let rgb = image::RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let [r, g, b, a] = rgba.get_pixel(x, y).0;
        let over = |c: u8| {
            let v = (u16::from(c) * u16::from(a) + 255 * (255 - u16::from(a)) + 127) / 255;
            u8::try_from(v).unwrap_or(u8::MAX)
        };
        image::Rgb([over(r), over(g), over(b)])
    });
    std::borrow::Cow::Owned(DynamicImage::ImageRgb8(rgb))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    struct Fake(std::result::Result<Vec<u8>, String>);

    impl Fetch for Fake {
        fn get(&self, _url: &str) -> std::result::Result<Vec<u8>, String> {
            self.0.clone()
        }
    }

    fn png() -> Vec<u8> {
        let mut out = Vec::new();
        DynamicImage::new_rgb8(4, 3)
            .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
            .unwrap();
        out
    }

    #[test]
    fn a_downloaded_png_is_sensed_with_its_title_and_format() {
        let s = Sensed::from_url("https://a.dev/img/chart.png?v=2", &Fake(Ok(png()))).unwrap();
        assert_eq!(s.title, "chart.png");
        assert_eq!(s.extension(), "png");
        assert_eq!(s.decode().unwrap().width(), 4);
    }

    #[test]
    fn an_html_page_is_not_an_image() {
        let err = Sensed::from_url("https://a.dev/", &Fake(Ok(b"<html>".to_vec()))).unwrap_err();
        assert!(matches!(err, Error::Decode(_)), "{err}");
    }

    #[test]
    fn a_failed_download_names_the_url() {
        let err =
            Sensed::from_url("https://a.dev/x.png", &Fake(Err("HTTP 404".into()))).unwrap_err();
        assert_eq!(err.to_string(), "GET https://a.dev/x.png: HTTP 404");
    }

    #[test]
    fn transparent_pixels_become_white_and_opaque_ones_stay() {
        let img = DynamicImage::ImageRgba8(image::RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                image::Rgba([0, 0, 0, 0])
            } else {
                image::Rgba([10, 20, 30, 255])
            }
        }));
        let flat = flatten(&img).to_rgb8();
        assert_eq!(flat.get_pixel(0, 0).0, [255, 255, 255]);
        assert_eq!(flat.get_pixel(1, 0).0, [10, 20, 30]);
    }

    #[test]
    fn a_windows_path_titles_by_file_name() {
        let s = Sensed::from_bytes(r"C:\shots\screen 1.png", png()).unwrap();
        assert_eq!(s.title, "screen 1.png");
    }
}
