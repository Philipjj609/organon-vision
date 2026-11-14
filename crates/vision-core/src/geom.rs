//! Regions in pixel coordinates of the original image: what every located observation points at.

use serde::{Deserialize, Serialize};

/// An axis-aligned rectangle in pixels, origin top-left. Serialized as `[x, y, w, h]`, the
/// shape of `Location::Page { bbox }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "[u32; 4]", into = "[u32; 4]")]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl From<[u32; 4]> for Region {
    fn from([x, y, w, h]: [u32; 4]) -> Self {
        Self { x, y, w, h }
    }
}

impl From<Region> for [u32; 4] {
    fn from(r: Region) -> Self {
        [r.x, r.y, r.w, r.h]
    }
}

impl Region {
    #[must_use]
    pub const fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    /// From edges `left, top, right, bottom` (right and bottom exclusive). Negative or inverted
    /// edges clamp to an empty region instead of wrapping.
    #[must_use]
    pub fn from_edges(left: i64, top: i64, right: i64, bottom: i64) -> Self {
        let clamp = |v: i64| u32::try_from(v.max(0)).unwrap_or(u32::MAX);
        let (l, t) = (clamp(left), clamp(top));
        Self::new(
            l,
            t,
            clamp(right).saturating_sub(l),
            clamp(bottom).saturating_sub(t),
        )
    }

    #[must_use]
    pub const fn right(&self) -> u32 {
        self.x + self.w
    }

    #[must_use]
    pub const fn bottom(&self) -> u32 {
        self.y + self.h
    }

    #[must_use]
    pub const fn area(&self) -> u64 {
        self.w as u64 * self.h as u64
    }

    /// The smallest region holding both.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let (x, y) = (self.x.min(other.x), self.y.min(other.y));
        let (r, b) = (
            self.right().max(other.right()),
            self.bottom().max(other.bottom()),
        );
        Self::new(x, y, r - x, b - y)
    }

    /// Shared area.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> u64 {
        let w = self
            .right()
            .min(other.right())
            .saturating_sub(self.x.max(other.x));
        let h = self
            .bottom()
            .min(other.bottom())
            .saturating_sub(self.y.max(other.y));
        u64::from(w) * u64::from(h)
    }

    /// Intersection over union, 0–1.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn iou(&self, other: &Self) -> f32 {
        let inter = self.intersection(other);
        let union = self.area() + other.area() - inter;
        if union == 0 {
            0.0
        } else {
            inter as f32 / union as f32
        }
    }

    /// Width of the shared horizontal span (0 when the columns do not meet).
    #[must_use]
    pub fn horizontal_overlap(&self, other: &Self) -> u32 {
        self.right()
            .min(other.right())
            .saturating_sub(self.x.max(other.x))
    }

    /// Grown by `by` pixels on every side (clamped at 0 on the top and left).
    #[must_use]
    pub fn grown(&self, by: u32) -> Self {
        let (x, y) = (self.x.saturating_sub(by), self.y.saturating_sub(by));
        Self::new(x, y, self.right() + by - x, self.bottom() + by - y)
    }

    /// `[x, y, w, h]` as protocol page units.
    #[must_use]
    pub fn to_bbox(&self) -> [f64; 4] {
        [self.x, self.y, self.w, self.h].map(f64::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_intersection_and_iou() {
        let a = Region::new(0, 0, 10, 10);
        let b = Region::new(5, 5, 10, 10);
        assert_eq!(a.union(&b), Region::new(0, 0, 15, 15));
        assert_eq!(a.intersection(&b), 25);
        assert!((a.iou(&b) - 25.0 / 175.0).abs() < 1e-6);
        assert_eq!(a.horizontal_overlap(&Region::new(20, 0, 5, 5)), 0);
    }

    #[test]
    fn edges_clamp_instead_of_wrapping() {
        assert_eq!(Region::from_edges(-3, -2, 7, 8), Region::new(0, 0, 7, 8));
        assert_eq!(Region::from_edges(9, 9, 4, 4).area(), 0);
    }

    #[test]
    fn serializes_as_bbox_array() {
        let r = Region::new(1, 2, 3, 4);
        assert_eq!(serde_json::to_string(&r).unwrap(), "[1,2,3,4]");
        assert_eq!(serde_json::from_str::<Region>("[1,2,3,4]").unwrap(), r);
    }
}
