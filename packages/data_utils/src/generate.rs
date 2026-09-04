//! # Generate
//! This module contains utilities for generating images with Cistercian numeral glyphs.
//! These will be used as training and evaluation data for the models, though it is better
//! to augment this with real data (drawings, handwritten examples). The glyphs will be grayscale
//! images represented as 2D arrays of floating point numbers between 0 (black) and 1 (white).

use crate::CResult;

use super::CistercianError;
use ndarray::Array2;

/// Relative tolerance for comparing floats
const RTOL: f32 = 1e-9;

/// Absolute tolerance for comparing floats
const ATOL: f32 = 1e-10;

/// Return whether two floats are approximately equal
pub fn approx_equal(a: f32, b: f32) -> bool {
    let rel_factor = a.abs().max(b.abs()) * RTOL;
    (a - b).abs() < ATOL + rel_factor
}

/// Positions in which a digit can be added to the numeral.
pub enum DigitPosition {
    Ones,
    Tens,
    Hundreds,
    Thousands,
}

/// Parameters to generate reference images of the Cistercian Numerals
pub struct ImageParams {
    /// Size of the image (Images are square)
    pub size: usize,

    /// Thickness of lines in the images
    pub thickness: usize,

    /// Space to leave as margin below / above the glyph
    pub bottom_margin: usize,
}

impl Default for ImageParams {
    /// Default Image Parameter set.
    fn default() -> Self {
        Self {
            size: 64,
            thickness: 2,
            bottom_margin: 4,
        }
    }
}

impl ImageParams {
    /// Get the Y-indices of the top and bottom pixels to draw.
    #[inline]
    pub fn top_bottom(&self) -> (usize, usize) {
        (self.bottom_margin, self.size - self.bottom_margin - 1)
    }

    /// Get the radius of the lines to draw
    #[inline]
    pub fn radius(&self) -> usize {
        (self.thickness / 2).max(1)
    }

    /// Return the middle height for drawing a digit in the given position
    pub fn middle_height(&self, position: DigitPosition) -> usize {
        let middle = self.size / 2;
        let quart = (middle - self.bottom_margin) / 2;
        match position {
            DigitPosition::Ones => middle - quart,
            DigitPosition::Tens => middle - quart,
            _ => middle + quart,
        }
    }
}

/// Generate an image with the given parameters and with the given number
/// in it. The image is generated as an array of floating point numbers between
/// 0 and 1.
pub fn generate_image(params: &ImageParams, digits: [u32; 4]) -> Array2<f32> {
    let mut img = Array2::ones((params.size, params.size));
    draw_digit(&mut img, params, digits[0], DigitPosition::Thousands);
    draw_digit(&mut img, params, digits[1], DigitPosition::Hundreds);
    draw_digit(&mut img, params, digits[2], DigitPosition::Tens);
    draw_digit(&mut img, params, digits[3], DigitPosition::Ones);
    img
}

/// Extract the 4 digits from a given number to encode in a glyph
pub fn number_to_digits(number: i32) -> CResult<[u32; 4]> {
    if number <= 0 || number > 9999 {
        return Err(CistercianError::InvalidNumber(number));
    }
    let mut remaining = number;
    let d1: u32 = (remaining % 10).cast_unsigned();
    remaining /= 10;
    let d2: u32 = (remaining % 10).cast_unsigned();
    remaining /= 10;
    let d3: u32 = (remaining % 10).cast_unsigned();
    remaining /= 10;
    let d4: u32 = (remaining % 10).cast_unsigned();

    Ok([d4, d3, d2, d1])
}

/// Draw a digit onto the image at the given position.
pub fn draw_digit(
    img: &mut Array2<f32>,
    params: &ImageParams,
    digit: u32,
    position: DigitPosition,
) {
}

#[cfg(test)]
mod tests {
    use std::assert_matches;

    use super::*;

    #[test]
    fn test_image_shape() {
        let digits = [1, 2, 3, 4];
        let mut params = ImageParams::default();

        let img = generate_image(&params, digits);
        assert_eq!(img.dim().0, params.size);
        assert_eq!(img.dim().1, params.size);

        params.size = 32;
        let img = generate_image(&params, digits);
        assert_eq!(img.dim().0, params.size);
        assert_eq!(img.dim().1, params.size);

        params.size = 72;
        let img = generate_image(&params, digits);
        assert_eq!(img.dim().0, params.size);
        assert_eq!(img.dim().1, params.size);
    }

    #[test]
    fn test_image_values() {
        let params = ImageParams::default();
        let img = generate_image(&params, [0, 1, 9, 1]);
        let (h, w) = img.dim();

        for i in 0..h {
            for j in 0..w {
                assert!(img[[i, j]] >= 0.0);
                assert!(img[[i, j]] <= 1.0);
            }
        }
    }

    #[test]
    fn test_number_conversion() {
        assert_eq!(number_to_digits(1234).unwrap(), [1, 2, 3, 4]);
        assert_eq!(number_to_digits(1).unwrap(), [0, 0, 0, 1]);
        assert_eq!(number_to_digits(23).unwrap(), [0, 0, 2, 3]);
        assert_eq!(number_to_digits(678).unwrap(), [0, 6, 7, 8]);
        assert_eq!(number_to_digits(1234).unwrap(), [1, 2, 3, 4]);
        assert_eq!(number_to_digits(9015).unwrap(), [9, 0, 1, 5]);
    }

    #[test]
    fn test_number_conversion_invalid() {
        assert_matches!(number_to_digits(0), Err(_));
        assert_matches!(number_to_digits(-12), Err(_));
        assert_matches!(number_to_digits(10000), Err(_));
        assert_matches!(number_to_digits(17056), Err(_));
        assert_matches!(number_to_digits(-2349), Err(_));
    }
}
