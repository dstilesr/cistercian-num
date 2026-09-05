use image::GrayImage;
use ndarray::Array2;

use super::super::{CResult, CistercianError};
use super::*;

/// Save an image (given as array) to a file.
pub fn save_to_file(image: Array2<u8>, filepath: &str) -> CResult<()> {
    log::debug!("Saving image to {}", filepath);
    let buffered = to_image(image)?;
    buffered
        .save(filepath)
        .map_err(|e| CistercianError::SerialisationError(e.to_string()))
}

/// Convert an array into a Grayscale image for serialisation.
pub fn to_image(data: Array2<u8>) -> CResult<GrayImage> {
    if !data.is_standard_layout() {
        return Err(CistercianError::SerialisationError(String::from(
            "Can only save if array is in standard layout!",
        )));
    }

    let (d1, d2) = data.dim();
    let (data_vec, _) = data.into_raw_vec_and_offset();
    let img = GrayImage::from_raw(d1 as u32, d2 as u32, data_vec).ok_or(
        CistercianError::SerialisationError(String::from("Could not instantiate image")),
    )?;
    Ok(img)
}
