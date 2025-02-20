use std::fmt::{Debug, Display, Formatter, Result};

#[derive(Debug)]
pub enum OperationError<T: Debug> {
    IndexOutOfBounds,
    UnsupportedOperation,
    DeviceError(Box<T>),
}

impl<T: Debug> Display for OperationError<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:?}", self)
    }
}

impl<T: Debug> std::error::Error for OperationError<T> {}

impl<T: Debug> From<T> for OperationError<T> {
    fn from(value: T) -> Self {
        Self::DeviceError(Box::new(value))
    }
}
