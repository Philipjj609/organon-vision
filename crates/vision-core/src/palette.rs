//! Dominant colours: k-means in CIE Lab (perceptual distances) with k-means++ seeding, via
//! `kmeans_colors`. The seed is fixed, so the same image always gives the same palette (and the
//! same observation id).

use image::DynamicImage;
use kmeans_colors::{get_kmeans_hamerly, Kmeans, Sort};
use palette::cast::from_component_slice;
use palette::{FromColor, IntoColor, Lab, Srgb};
use serde::{Deserialize, Serialize};

/// One colour of the palette.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Swatch {
    /// `#rrggbb`.
    pub hex: String,
    /// Share of the image's pixels nearest this colour, 0–1, three decimals.
    pub share: f64,
}

/// How the palette is computed.
#[derive(Debug, Clone, Copy)]
pub struct PaletteRules {
    /// Colours to look for.
    pub k: usize,
    /// Longest side the image is shrunk to first; the palette does not need every pixel.
    pub max_side: u32,
    /// Colours closer than this (CIE76 ΔE) are merged: two clusters of one flat background
    /// are one colour.
    pub merge_delta_e: f32,
    /// Colours below this share are dropped.
    pub min_share: f64,
}

impl Default for PaletteRules {
    fn default() -> Self {
        Self {
            k: 8,
            max_side: 200,
            merge_delta_e: 8.0,
            min_share: 0.01,
        }
    }
}

const SEED: u64 = 0x6f72_67616e;
const MAX_ITER: usize = 30;
const CONVERGE: f32 = 0.5;

/// The dominant colours, most frequent first.
#[must_use]
pub fn dominant(img: &DynamicImage, rules: PaletteRules) -> Vec<Swatch> {
    let small = img.thumbnail(rules.max_side, rules.max_side).to_rgb8();
    let lab: Vec<Lab> = from_component_slice::<Srgb<u8>>(small.as_raw())
        .iter()
        .map(|c| c.into_linear::<f32>().into_color())
        .collect();
    if lab.is_empty() {
        return Vec::new();
    }
    let k = rules.k.clamp(1, lab.len());
    let result: Kmeans<Lab> = get_kmeans_hamerly(k, MAX_ITER, CONVERGE, false, &lab, SEED);
    let mut clusters: Vec<(Lab, f32)> =
        Lab::sort_indexed_colors(&result.centroids, &result.indices)
            .into_iter()
            .map(|c| (c.centroid, c.percentage))
            .collect();
    clusters.sort_by(|a, b| b.1.total_cmp(&a.1));
    merge_close(&mut clusters, rules.merge_delta_e);
    clusters
        .into_iter()
        .map(|(c, share)| Swatch {
            hex: hex(c),
            share: round3(f64::from(share)),
        })
        .filter(|s| s.share >= rules.min_share)
        .collect()
}

/// Fold each colour into an earlier (more frequent) one within `delta_e`.
fn merge_close(clusters: &mut Vec<(Lab, f32)>, delta_e: f32) {
    let mut kept: Vec<(Lab, f32)> = Vec::with_capacity(clusters.len());
    for &(c, share) in clusters.iter() {
        match kept.iter_mut().find(|(k, _)| distance(*k, c) < delta_e) {
            Some(k) => k.1 += share,
            None => kept.push((c, share)),
        }
    }
    kept.sort_by(|a, b| b.1.total_cmp(&a.1));
    *clusters = kept;
}

fn distance(a: Lab, b: Lab) -> f32 {
    ((a.l - b.l).powi(2) + (a.a - b.a).powi(2) + (a.b - b.b).powi(2)).sqrt()
}

fn hex(c: Lab) -> String {
    let rgb: Srgb<u8> = Srgb::from_linear(palette::LinSrgb::from_color(c));
    format!("#{:02x}{:02x}{:02x}", rgb.red, rgb.green, rgb.blue)
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn img(f: impl Fn(u32, u32) -> [u8; 3]) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(120, 80, |x, y| Rgb(f(x, y))))
    }

    #[test]
    fn two_flat_colours_give_two_swatches_by_share() {
        // three quarters white, one quarter red
        let p = dominant(
            &img(|x, _| {
                if x < 90 {
                    [255, 255, 255]
                } else {
                    [200, 20, 20]
                }
            }),
            PaletteRules::default(),
        );
        assert_eq!(p.len(), 2, "{p:?}");
        assert_eq!(p[0].hex, "#ffffff");
        assert!((p[0].share - 0.75).abs() < 0.02, "{p:?}");
        assert_eq!(p[1].hex, "#c81414");
    }

    #[test]
    fn a_single_colour_is_one_swatch_not_k_copies() {
        let p = dominant(&img(|_, _| [10, 20, 30]), PaletteRules::default());
        assert_eq!(p.len(), 1, "{p:?}");
        assert_eq!(p[0].share, 1.0);
    }

    #[test]
    fn the_same_image_gives_the_same_palette() {
        let pic = img(|x, y| [(x * 2) as u8, (y * 3) as u8, ((x + y) % 256) as u8]);
        assert_eq!(
            dominant(&pic, PaletteRules::default()),
            dominant(&pic, PaletteRules::default())
        );
    }
}
