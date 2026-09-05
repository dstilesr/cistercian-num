use std::{error::Error, fmt::Display};

pub mod generate;

#[cfg(feature = "python_binding")]
pub mod python_binding;

/// Error class for the library
#[derive(Debug)]
pub enum CistercianError {
    InvalidNumber(i32),
    InvalidConfig(String),
    SerialisationError(String),
    Fail,
}

impl Display for CistercianError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::InvalidNumber(n) => write!(f, "Cannot encode the given number {}", n),
            Self::InvalidConfig(s) => write!(f, "{s}"),
            Self::Fail => write!(f, "Failed to perform operation!"),
            Self::SerialisationError(s) => write!(f, "{s}"),
        }
    }
}

impl Error for CistercianError {}

/// Result type for the library
pub type CResult<T> = Result<T, CistercianError>;
