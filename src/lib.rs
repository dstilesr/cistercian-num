use std::{error::Error, fmt::Display};

use crate::CistercianError::InvalidNumber;

pub mod generate;

#[cfg(feature = "python_binding")]
pub mod python_binding;

/// Result type for the library
pub type CResult<T> = Result<T, CistercianError>;

/// Error class for the library
#[derive(Debug)]
pub enum CistercianError {
    InvalidNumber(i32),
}

impl Display for CistercianError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            InvalidNumber(n) => write!(f, "Cannot encode the given number {}", n),
        }
    }
}

impl Error for CistercianError {}
