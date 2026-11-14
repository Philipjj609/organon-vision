//! Text in an image, as lines with their regions. Reading is an engine's job (see
//! [`crate::engines`]); this module only says what a reader returns.

use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::geom::Region;

/// One recognised word.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub text: String,
    pub region: Region,
}

/// One recognised line of text, left to right.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextLine {
    pub text: String,
    pub region: Region,
    pub words: Vec<Word>,
}

impl TextLine {
    /// A line from words; its region is the union of theirs. `None` for no words.
    #[must_use]
    pub fn from_words(words: Vec<Word>) -> Option<Self> {
        let region = words.iter().map(|w| w.region).reduce(|a, b| a.union(&b))?;
        let text = words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Some(Self {
            text,
            region,
            words,
        })
    }
}

/// Reads the text of an image. Implemented by OCR engines; tests use a fake.
pub trait TextReader {
    /// Engine and version, recorded as the observation's `produced_by.tool` (`ocrs 0.14`).
    fn tool(&self) -> String;

    /// Lines of text, in any order.
    ///
    /// # Errors
    /// The engine failed on this image.
    fn read(&self, img: &DynamicImage) -> Result<Vec<TextLine>>;
}

/// Word gap, in word heights, above which one engine line is really two (side-by-side boxes of
/// a diagram, columns, a status bar). Normal word spacing is under half a height.
pub const SPLIT_GAP: f32 = 2.0;

/// Clean an engine's lines: split each at wide word gaps (ocrs joins words across columns of a
/// diagram into one line: "kubelet  kubelet  kubelet"), then drop what engines commonly
/// hallucinate on texture: empty lines and lone characters.
#[must_use]
pub fn tidy(lines: Vec<TextLine>) -> Vec<TextLine> {
    lines
        .into_iter()
        .flat_map(split_at_gaps)
        .filter(|l| l.text.trim().chars().count() > 1)
        .collect()
}

/// Split a line wherever the gap between neighbouring words exceeds [`SPLIT_GAP`] times the
/// taller of the two. A line without words is kept as it is.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn split_at_gaps(line: TextLine) -> Vec<TextLine> {
    if line.words.len() < 2 {
        return vec![line];
    }
    let mut words = line.words;
    words.sort_by_key(|w| w.region.x);
    let mut groups: Vec<Vec<Word>> = vec![Vec::new()];
    for w in words {
        let current = groups.last_mut().expect("groups starts non-empty");
        let wide = current.last().is_some_and(|prev| {
            let gap = w.region.x.saturating_sub(prev.region.right()) as f32;
            gap > SPLIT_GAP * prev.region.h.max(w.region.h) as f32
        });
        if wide {
            groups.push(vec![w]);
        } else {
            current.push(w);
        }
    }
    groups
        .into_iter()
        .filter_map(TextLine::from_words)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Region;

    fn word(text: &str, x: u32) -> Word {
        Word {
            text: text.to_owned(),
            region: Region::new(x, 100, 60, 20),
        }
    }

    /// Regression (real run on the Kubernetes components diagram): three node boxes read as one
    /// line "kubelet kubelet kubelet".
    #[test]
    fn words_far_apart_are_separate_lines() {
        let words = vec![
            word("kubelet", 0),
            word("kubelet", 300),
            word("kubelet", 600),
        ];
        let split = tidy(vec![TextLine::from_words(words).unwrap()]);
        assert_eq!(split.len(), 3, "{split:?}");
        assert_eq!(split[1].region, Region::new(300, 100, 60, 20));
    }

    #[test]
    fn normal_word_spacing_keeps_one_line() {
        let line = TextLine::from_words(vec![word("Load", 0), word("Balancer", 68)]).unwrap();
        let split = tidy(vec![line]);
        assert_eq!(split.len(), 1);
        assert_eq!(split[0].text, "Load Balancer");
    }

    #[test]
    fn lone_characters_are_dropped() {
        let line = TextLine::from_words(vec![word("x", 0)]).unwrap();
        assert!(tidy(vec![line]).is_empty());
    }
}
