mod binary;
mod matmul;
mod sparse;
mod trinary;
mod unary;

pub use binary::*;
pub use matmul::*;
pub use sparse::*;
pub use trinary::*;
pub use unary::*;

use crate::{
    device::{Device, OperationError},
    graph::{Graph, GraphNodeId},
};

pub trait GraphInstruction<D: Device>: std::fmt::Debug + 'static {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>>;
}

#[derive(Debug)]
pub struct Set {
    pub id: GraphNodeId,
    pub val: f32,
}

impl<D: Device> GraphInstruction<D> for Set {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        graph.get_mut(self.id)?.dense_mut()?.set_to(self.val)?;

        Ok(())
    }
}
