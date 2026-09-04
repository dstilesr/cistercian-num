use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::CistercianError::InvalidNumber;
use crate::generate::generate_image;

use super::generate;
use super::{CResult, CistercianError};

impl CistercianError {
    /// Map package Errors into Python error types
    pub fn to_python_error(self) -> PyErr {
        match self {
            InvalidNumber(_) => PyValueError::new_err(format!("{}", self)),
        }
    }
}

/// Generate an image of a glyph with a Cistercian Numeral
#[pyfunction]
pub fn make_image(
    py: Python,
    size: usize,
    line_thickness: usize,
    margin: usize,
    number: i32,
) -> PyResult<Bound<PyArray2<f32>>> {
    let params = generate::ImageParams {
        size: size,
        thickness: line_thickness,
        bottom_margin: margin,
    };
    let digits = generate::number_to_digits(number).map_err(|e| e.to_python_error())?;
    let img = generate::generate_image(&params, digits);

    Ok(PyArray2::from_array(py, &img))
}

#[pyo3::pymodule]
#[pyo3(name = "cistercian")]
pub mod cistercian {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::make_image;

    /// Module initialization - setup Python logging integration
    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        pyo3_log::init();
        Ok(())
    }
}
