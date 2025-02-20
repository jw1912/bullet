use std::fmt::Debug;

use crate::device::OperationError;

use super::operation::GraphBuilderError;

#[derive(Debug)]
pub enum GraphError<T: Debug> {
    Builder(GraphBuilderError),
    Operation(OperationError<T>),
    DeviceError(T),
}

impl<T: Debug> From<GraphBuilderError> for GraphError<T> {
    fn from(value: GraphBuilderError) -> Self {
        Self::Builder(value)
    }
}

impl<T: Debug> From<OperationError<T>> for GraphError<T> {
    fn from(value: OperationError<T>) -> Self {
        Self::Operation(value)
    }
}
