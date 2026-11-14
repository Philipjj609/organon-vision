//! Layout blocks: where the text paragraphs and the figures are. Deterministic geometry, no
//! model: text lines are grouped into blocks by alignment and spacing, and the busy parts of the
//! image that hold no text become figure blocks.
//!
//! A model (DocLayout-YOLO, PP-DocLayout) would name blocks (title, table, caption); that is a
//! later engine behind the same [`Block`] shape. See docs/DESIGN.md.

use std::collections::VecDeque;

use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::geom::Region;
use crate::measure::{Features, Look, LookRules};
use crate::text::TextLine;

/// What a block holds. Closed set: measured, not interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// One or more aligned, evenly spaced lines of text.
    Text,
    /// A busy region without text: a picture, a chart, a drawing, a photo.
    Figure,
}

/// A layout block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub role: Role,
    pub region: Region,
    /// Text blocks: positions of their lines in the line list, top to bottom.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lines: Vec<usize>,
    /// Text blocks: median line height in pixels (a typographic size).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_height: Option<u32>,
    /// Figure blocks: whether the crop looks like strokes (`graphic`) or a photo (`scene`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub look: Option<Look>,
}

/// Group lines into text blocks. A line joins the nearest block above it when their columns
/// overlap, the gap to the block's last line is at most 1.2 line heights, and the two line
/// heights differ by at most 1.6 times.
#[must_use]
pub fn text_blocks(lines: &[TextLine]) -> Vec<Block> {
    let mut order: Vec<usize> = (0..lines.len()).collect();
    order.sort_by_key(|&i| (lines[i].region.y, lines[i].region.x));
    let mut blocks: Vec<(Region, Vec<usize>)> = Vec::new();
    for i in order {
        let r = lines[i].region;
        let h = r.h.max(1);
        let joins = |(b, members): &(Region, Vec<usize>)| {
            let last = lines[*members.last().expect("blocks are never empty")].region;
            let (lo, hi) = (h.min(last.h.max(1)), h.max(last.h.max(1)));
            let gap = i64::from(r.y) - i64::from(last.bottom());
            let column = b.horizontal_overlap(&r) > 0 || b.x.abs_diff(r.x) <= h;
            column && gap >= -i64::from(lo / 2) && gap <= i64::from(lo) * 6 / 5 && hi * 5 <= lo * 8
        };
        let target = blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| joins(b))
            .max_by_key(|(_, b)| b.0.bottom())
            .map(|(k, _)| k);
        match target {
            Some(k) => {
                blocks[k].0 = blocks[k].0.union(&r);
                blocks[k].1.push(i);
            }
            None => blocks.push((r, vec![i])),
        }
    }
    blocks
        .into_iter()
        .map(|(region, members)| {
            let mut heights: Vec<u32> = members.iter().map(|&i| lines[i].region.h).collect();
            heights.sort_unstable();
            Block {
                role: Role::Text,
                region,
                line_height: Some(heights[heights.len() / 2]),
                lines: members,
                look: None,
            }
        })
        .collect()
}

/// Longest side of the working copy for figure detection.
const WORK_SIDE: u32 = 512;
/// Tile side in working pixels.
const TILE: u32 = 16;
/// Luma step (|dx| + |dy|) that makes an edge pixel.
const EDGE: i16 = 40;
/// Share of edge pixels that makes a tile busy.
const BUSY: f32 = 0.05;
/// A figure fills at least this share of its bounding box with busy tiles (the thin border of a
/// panel does not), and covers at least `MIN_AREA` of the image.
const MIN_FILL: f32 = 0.35;
const MIN_AREA: f32 = 0.01;
const MIN_TILES: usize = 4;

/// Busy regions that hold no text, as figure blocks.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn figure_blocks(img: &DynamicImage, text: &[Region]) -> Vec<Block> {
    let (w0, h0) = (img.width(), img.height());
    if w0 < TILE || h0 < TILE {
        return Vec::new();
    }
    let scale = (WORK_SIDE as f32 / w0.max(h0) as f32).min(1.0);
    let g = img
        .resize_exact(
            ((w0 as f32 * scale) as u32).max(TILE),
            ((h0 as f32 * scale) as u32).max(TILE),
            image::imageops::FilterType::Triangle,
        )
        .to_luma8();
    let (tw, th) = (g.width() / TILE, g.height() / TILE);
    let to_work = |r: &Region| {
        let s = |v: u32| (v as f32 * scale) as u32;
        Region::new(s(r.x), s(r.y), s(r.w), s(r.h)).grown(TILE / 4)
    };
    let text_work: Vec<Region> = text.iter().map(to_work).collect();
    let at = |x: u32, y: u32| i16::from(g.get_pixel(x, y)[0]);
    let mut busy = vec![false; (tw * th) as usize];
    for ty in 0..th {
        for tx in 0..tw {
            let tile = Region::new(tx * TILE, ty * TILE, TILE, TILE);
            let covered: u64 = text_work.iter().map(|t| t.intersection(&tile)).sum();
            if covered * 10 >= tile.area() * 3 {
                continue;
            }
            let mut edges = 0_u32;
            for y in tile.y..tile.bottom() - 1 {
                for x in tile.x..tile.right() - 1 {
                    let c = at(x, y);
                    if (at(x + 1, y) - c).abs() + (at(x, y + 1) - c).abs() > EDGE {
                        edges += 1;
                    }
                }
            }
            busy[(ty * tw + tx) as usize] = edges as f32 >= BUSY * ((TILE - 1) * (TILE - 1)) as f32;
        }
    }
    let total_tiles = (tw * th) as f32;
    let mut seen = vec![false; busy.len()];
    let mut out = Vec::new();
    for start in 0..busy.len() {
        if !busy[start] || seen[start] {
            continue;
        }
        let (count, (x0, y0, x1, y1)) = component(&busy, &mut seen, start, tw, th);
        let box_tiles = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f32;
        if count < MIN_TILES
            || (count as f32) < MIN_FILL * box_tiles
            || box_tiles < MIN_AREA * total_tiles
        {
            continue;
        }
        let back = |v: u32| ((v * TILE) as f32 / scale).round() as i64;
        let region = Region::from_edges(
            back(x0),
            back(y0),
            back(x1 + 1).min(i64::from(w0)),
            back(y1 + 1).min(i64::from(h0)),
        );
        let crop = img.crop_imm(region.x, region.y, region.w, region.h);
        out.push(Block {
            role: Role::Figure,
            region,
            lines: Vec::new(),
            line_height: None,
            look: Some(Features::of(&crop).look(LookRules::default())),
        });
    }
    drop_nested(out)
}

/// Drop figures lying (90 % or more) inside a larger one: components are disjoint, their
/// bounding boxes are not (a strip of sky and the whole photo).
fn drop_nested(mut figures: Vec<Block>) -> Vec<Block> {
    figures.sort_by_key(|b| std::cmp::Reverse(b.region.area()));
    let mut kept: Vec<Block> = Vec::with_capacity(figures.len());
    for f in figures {
        let inside = kept
            .iter()
            .any(|k| k.region.intersection(&f.region) * 10 >= f.region.area() * 9);
        if !inside {
            kept.push(f);
        }
    }
    kept
}

/// Flood-fill one 4-connected component of busy tiles: (tile count, (x0, y0, x1, y1)).
fn component(
    busy: &[bool],
    seen: &mut [bool],
    start: usize,
    tw: u32,
    th: u32,
) -> (usize, (u32, u32, u32, u32)) {
    let idx = |x: u32, y: u32| (y * tw + x) as usize;
    let start_u32 = u32::try_from(start).expect("tile index fits u32: tw * th does");
    let (sx, sy) = (start_u32 % tw, start_u32 / tw);
    let mut bbox = (sx, sy, sx, sy);
    let mut queue = VecDeque::from([(sx, sy)]);
    seen[start] = true;
    let mut count = 0;
    while let Some((x, y)) = queue.pop_front() {
        count += 1;
        bbox = (bbox.0.min(x), bbox.1.min(y), bbox.2.max(x), bbox.3.max(y));
        let near = [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ];
        for (nx, ny) in near {
            if nx < tw && ny < th && busy[idx(nx, ny)] && !seen[idx(nx, ny)] {
                seen[idx(nx, ny)] = true;
                queue.push_back((nx, ny));
            }
        }
    }
    (count, bbox)
}

/// Text blocks, then figure blocks, in reading order (top to bottom, then left to right).
#[must_use]
pub fn blocks(img: &DynamicImage, lines: &[TextLine]) -> Vec<Block> {
    let regions: Vec<Region> = lines.iter().map(|l| l.region).collect();
    let mut all = text_blocks(lines);
    all.extend(figure_blocks(img, &regions));
    all.sort_by_key(|b| (b.region.y, b.region.x));
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn line(x: u32, y: u32, w: u32, h: u32) -> TextLine {
        TextLine {
            text: format!("line at {y}"),
            region: Region::new(x, y, w, h),
            words: Vec::new(),
        }
    }

    #[test]
    fn a_paragraph_is_one_block_and_a_far_heading_another() {
        let lines = [
            line(40, 20, 300, 40), // big heading
            line(40, 120, 500, 16),
            line(40, 140, 480, 16),
            line(40, 160, 300, 16),
        ];
        let b = text_blocks(&lines);
        assert_eq!(b.len(), 2, "{b:?}");
        assert_eq!(b[0].lines, [0]);
        assert_eq!(b[0].line_height, Some(40));
        assert_eq!(b[1].lines, [1, 2, 3]);
        assert_eq!(b[1].region, Region::new(40, 120, 500, 56));
    }

    #[test]
    fn two_columns_stay_two_blocks() {
        let lines = [
            line(20, 100, 200, 16),
            line(400, 100, 200, 16),
            line(20, 120, 200, 16),
            line(400, 120, 200, 16),
        ];
        let b = text_blocks(&lines);
        assert_eq!(b.len(), 2, "{b:?}");
        assert_eq!(b[0].lines, [0, 2]);
        assert_eq!(b[1].lines, [1, 3]);
    }

    fn page(f: impl Fn(u32, u32) -> u8) -> DynamicImage {
        DynamicImage::ImageLuma8(GrayImage::from_fn(512, 512, |x, y| Luma([f(x, y)])))
    }

    #[test]
    fn a_line_drawing_on_a_blank_page_is_one_graphic_figure() {
        let img = page(|x, y| {
            let inside = (300..460).contains(&x) && (40..200).contains(&y);
            if inside && (x % 5 == 0 || y % 7 == 0) {
                20
            } else {
                250
            }
        });
        let f = figure_blocks(&img, &[]);
        assert_eq!(f.len(), 1, "{f:?}");
        let r = f[0].region;
        assert!(
            r.x <= 300 && r.right() >= 448 && r.y <= 48 && r.bottom() >= 192,
            "{r:?}"
        );
        assert_eq!(f[0].look, Some(Look::Graphic));
    }

    /// Regression (real run on a photo of a bus): a strip of sky came out as a second figure
    /// inside the whole-photo figure.
    #[test]
    fn a_figure_inside_another_is_dropped() {
        let fig = |x, y, w, h| Block {
            role: Role::Figure,
            region: Region::new(x, y, w, h),
            lines: Vec::new(),
            line_height: None,
            look: None,
        };
        let kept = drop_nested(vec![
            fig(304, 0, 270, 68),
            fig(0, 0, 810, 1080),
            fig(900, 0, 50, 50),
        ]);
        let regions: Vec<Region> = kept.iter().map(|b| b.region).collect();
        assert_eq!(
            regions,
            [Region::new(0, 0, 810, 1080), Region::new(900, 0, 50, 50)]
        );
    }

    #[test]
    fn a_panel_border_and_known_text_are_not_figures() {
        // a 1-px rectangle outline (a UI panel) and a striped area already read as text
        let img = page(|x, y| {
            let border = ((x == 50 || x == 450) && (50..450).contains(&y))
                || ((y == 50 || y == 450) && (50..450).contains(&x));
            let striped = (100..300).contains(&x) && (100..140).contains(&y) && x % 3 == 0;
            if border || striped {
                0
            } else {
                255
            }
        });
        let f = figure_blocks(&img, &[Region::new(100, 100, 200, 40)]);
        assert!(f.is_empty(), "{f:?}");
    }
}
