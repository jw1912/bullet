use std::fmt::Debug;

#[derive(Debug)]
pub enum OperationError<T: Debug> {
    TensorOptimisedOut,
    InvalidTensorFormat,
    IndexOutOfBounds,
    UnsupportedOperation,
    MismatchedBatchSizes,
    DeviceError(Box<T>),
}

impl<T: Debug> From<T> for OperationError<T> {
    fn from(value: T) -> Self {
        Self::DeviceError(Box::new(value))
    }
}

pub type OperationResult<T> = Result<(), OperationError<T>>;
