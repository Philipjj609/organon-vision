//! Whole-image measures: what an image looks like, measured, not interpreted. A difference hash
//! finds repeats (a logo shown six times is one picture); brightness finds dark frames; "ink"
//! (thin strokes, in the densest part of the image) hints at text, code and drawings, even when
//! they fill only a corner: a placard, a monitor, a whiteboard.
//!
//! Moved here from organon-video's `frames` module unchanged (same thresholds, same numbers), so
//! video can measure its keyframes by calling this crate. `tests::golden_*` pins the numbers.
use serde::{Deserialize, Serialize};

/// Measured features of one frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Features {
    /// Mean brightness 0–255.
    pub mean: f32,
    /// Share of near-white pixels (slides, documents, whiteboards).
    pub white: f32,
    /// Thin-stroke density in the densest twelfth-or-so of the frame (0–1): strokes are pixels
    /// darker (or lighter) than both neighbours two pixels away, which text and line drawings
    /// have and photographs mostly do not. A hint, not a verdict.
    pub ink: f32,
    /// 64-bit difference hash: similar pictures have a small Hamming distance.
    pub dhash: u64,
}

/// Coarse appearance, from features alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Look {
    /// Black or nearly black: transitions, fades.
    Dark,
    /// Dense thin strokes somewhere: likely text, code, a diagram or a drawing. Worth reading.
    Graphic,
    /// A camera shot.
    Scene,
}

/// Thresholds for [`Look`]. Tuned on a real talk with hand-labelled frames (docs/DESIGN.md):
/// at `graphic_ink` 0.2, 12 of 14 flagged frames held code or diagrams; faint whiteboards are missed.
#[derive(Debug, Clone, Copy)]
pub struct LookRules {
    pub dark_mean: f32,
    pub graphic_ink: f32,
}

impl Default for LookRules {
    fn default() -> Self {
        Self {
            dark_mean: 18.0,
            graphic_ink: 0.2,
        }
    }
}

impl Features {
    /// Measure an image.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn of(img: &image::DynamicImage) -> Self {
        let small = img
            .resize_exact(160, 90, image::imageops::FilterType::Triangle)
            .to_luma8();
        let px = small.as_raw();
        let n = px.len() as f32;
        let mean = px.iter().map(|&p| f32::from(p)).sum::<f32>() / n;
        let white = px.iter().filter(|&&p| p > 200).count() as f32 / n;
        Self {
            mean,
            white,
            ink: ink(img),
            dhash: dhash(img),
        }
    }

    /// Measure an encoded image (JPEG or PNG).
    ///
    /// # Errors
    /// The bytes are not a decodable image.
    pub fn of_encoded(bytes: &[u8]) -> Result<Self, image::ImageError> {
        Ok(Self::of(&image::load_from_memory(bytes)?))
    }

    #[must_use]
    pub fn look(&self, rules: LookRules) -> Look {
        if self.mean < rules.dark_mean && self.ink < 0.01 {
            Look::Dark
        } else if self.ink >= rules.graphic_ink {
            Look::Graphic
        } else {
            Look::Scene
        }
    }
}

const INK_W: u32 = 480;
const INK_H: u32 = 270;
const TILE: u32 = 30;
/// Tiles averaged for the score: the densest ~8% of the frame.
const TOP_TILES: usize = 12;
/// Brightness step that makes a stroke.
const STROKE: i16 = 25;

/// Thin-stroke density of the densest tiles.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn ink(img: &image::DynamicImage) -> f32 {
    let g = img
        .resize_exact(INK_W, INK_H, image::imageops::FilterType::Triangle)
        .to_luma8();
    let at = |x: u32, y: u32| i16::from(g.get_pixel(x, y)[0]);
    let stroke = |c: i16, a: i16, b: i16| {
        (a - c > STROKE && b - c > STROKE) || (c - a > STROKE && c - b > STROKE)
    };
    let inner = (TILE - 4) * (TILE - 4);
    let mut tiles: Vec<f32> = Vec::with_capacity(((INK_W / TILE) * (INK_H / TILE)) as usize);
    for ty in 0..INK_H / TILE {
        for tx in 0..INK_W / TILE {
            let mut count = 0_u32;
            for y in ty * TILE + 2..ty * TILE + TILE - 2 {
                for x in tx * TILE + 2..tx * TILE + TILE - 2 {
                    let c = at(x, y);
                    if stroke(c, at(x - 2, y), at(x + 2, y))
                        || stroke(c, at(x, y - 2), at(x, y + 2))
                    {
                        count += 1;
                    }
                }
            }
            tiles.push(count as f32 / inner as f32);
        }
    }
    tiles.sort_by(|a, b| b.total_cmp(a));
    let top = &tiles[..TOP_TILES.min(tiles.len())];
    top.iter().sum::<f32>() / top.len().max(1) as f32
}

/// Difference hash: shrink to 9×8 grey, one bit per "left brighter than right".
#[must_use]
pub fn dhash(img: &image::DynamicImage) -> u64 {
    let g = img
        .resize_exact(9, 8, image::imageops::FilterType::Triangle)
        .to_luma8();
    let mut bits = 0_u64;
    for y in 0..8 {
        for x in 0..8 {
            bits <<= 1;
            if g.get_pixel(x, y)[0] > g.get_pixel(x + 1, y)[0] {
                bits |= 1;
            }
        }
    }
    bits
}

#[must_use]
pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};

    fn image(f: impl Fn(u32, u32) -> u8) -> DynamicImage {
        DynamicImage::ImageLuma8(GrayImage::from_fn(320, 180, |x, y| Luma([f(x, y)])))
    }

    fn look(img: &DynamicImage) -> Look {
        Features::of(img).look(LookRules::default())
    }

    #[test]
    fn black_is_dark_text_is_graphic_gradient_is_scene() {
        let black = image(|_, _| 3);
        // white page with dense thin dark strokes, like small print
        let text = image(|x, y| if x % 4 == 0 || y % 6 == 0 { 20 } else { 250 });
        let gradient = image(|x, _| (x * 255 / 320) as u8);
        assert_eq!(look(&black), Look::Dark);
        assert_eq!(look(&text), Look::Graphic);
        assert_eq!(look(&gradient), Look::Scene);
    }

    #[test]
    fn a_small_placard_of_text_in_a_camera_shot_is_graphic() {
        // Smooth "room" (gradients, soft blobs) with a white card of thin strokes in one corner,
        // like a diagram held up to the camera: about a tenth of the frame.
        let room = |x: u32, y: u32| (60 + (x * 90 / 320) + (y * 40 / 180)) as u8;
        let card = |x: u32, y: u32| (210..300).contains(&x) && (110..170).contains(&y);
        let shot = image(|x, y| match card(x, y) {
            true if x % 4 == 0 || y % 7 == 0 => 30,
            true => 245,
            false => room(x, y),
        });
        assert_eq!(
            look(&shot),
            Look::Graphic,
            "ink {}",
            Features::of(&shot).ink
        );
        assert_eq!(look(&image(room)), Look::Scene);
    }

    #[test]
    fn the_same_picture_hashes_close_and_a_different_one_far() {
        let logo = image(|x, y| if (x / 40 + y / 40) % 2 == 0 { 230 } else { 30 });
        let other = image(|x, _| (x * 255 / 320) as u8);
        assert_eq!(hamming(dhash(&logo), dhash(&logo)), 0);
        assert!(hamming(dhash(&logo), dhash(&other)) > 6);
    }

    /// Pins the numbers organon-video's ids depend on: if this changes, video's observation
    /// ids change with it.
    #[test]
    fn golden_measures_of_a_fixed_pattern() {
        let pattern = image(|x, y| ((x * 7 + y * 13 + (x * y) / 50) % 256) as u8);
        let f = Features::of(&pattern);
        assert_eq!(
            GOLDEN,
            (
                f.mean.to_bits(),
                f.white.to_bits(),
                f.ink.to_bits(),
                f.dhash
            ),
            "{f:?}"
        );
    }

    /// Checked equal to organon-video 0.1 `frames::Features::of` on 2026-11-14.
    const GOLDEN: (u32, u32, u32, u64) = (
        1_124_018_859,
        1_041_767_247,
        1_031_947_694,
        9_804_987_581_031_686_416,
    );
}
