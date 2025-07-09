use crate::{
    backend::device::{Device, OperationError},
    graph::Graph,
};

pub trait GraphInstruction<D: Device>: 'static {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>>;
}
