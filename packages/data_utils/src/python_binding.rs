use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::RngExt;

use super::CistercianError;
use super::generate;

impl CistercianError {
    /// Map package Errors into Python error types
    pub fn to_python_error(self) -> PyErr {
        match self {
            Self::InvalidConfig(s) => PyValueError::new_err(s),
            _ => PyValueError::new_err(format!("{}", self)),
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
) -> PyResult<Bound<PyArray2<u8>>> {
    let params = generate::ImageParams {
        size: size,
        thickness: line_thickness,
        bottom_margin: margin,
    };
    let digits = generate::number_to_digits(number).map_err(|e| e.to_python_error())?;
    let img = generate::generate_image(&params, digits).map_err(|e| e.to_python_error())?;

    Ok(PyArray2::from_array(py, &img))
}

/// Generate an image of a numeral and save it at the given filepath.
#[pyfunction]
pub fn generate_and_save_image(
    size: usize,
    line_thickness: usize,
    margin: usize,
    number: i32,
    filepath: String,
) -> PyResult<()> {
    let params = generate::ImageParams {
        size: size,
        thickness: line_thickness,
        bottom_margin: margin,
    };
    let digits = generate::number_to_digits(number).map_err(|e| e.to_python_error())?;
    let img = generate::generate_image(&params, digits).map_err(|e| e.to_python_error())?;
    generate::serialise::save_to_file(img, &filepath).map_err(|e| e.to_python_error())
}

/// Generate two lists of numbers to use for training and test, respectively.
#[pyfunction]
#[pyo3(signature = (test_proportion: "float") -> "tuple[list[int], list[int]]")]
pub fn train_test_numbers(test_proportion: f32) -> PyResult<(Vec<u32>, Vec<u32>)> {
    if test_proportion > 1.0 || test_proportion < 0.0 {
        return Err(PyValueError::new_err(format!(
            "Proportion of test set elements must be between 0 and 1. Got {}",
            test_proportion
        )));
    }
    log::info!(
        "Splitting numbers into train - test (test proportion={})",
        test_proportion
    );
    let mut random_gen = rand::rng();
    let total = 10000;

    let total_test = (test_proportion * 10000.0).round() as usize;
    let mut train = Vec::with_capacity(total - total_test + 32);
    let mut test = Vec::with_capacity(total_test + 32);

    for i in 1..10000 {
        let draw: f32 = random_gen.random();
        if draw < test_proportion {
            test.push(i);
        } else {
            train.push(i);
        }
    }

    Ok((train, test))
}

#[pyo3::pymodule]
#[pyo3(name = "cistercian")]
pub mod cistercian {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::{generate_and_save_image, make_image, train_test_numbers};

    /// Module initialization - setup Python logging integration
    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        pyo3_log::init();
        Ok(())
    }
}
