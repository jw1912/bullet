mod matmul;
mod sparse;

pub use matmul::Matmul;
pub use sparse::{BackpropSparseAffineActivateStrided, SparseAffineActivateStrided};

use crate::{
    backend::device::{Device, OperationError},
    graph::{Graph, NodeId},
};

pub trait GraphInstruction<D: Device>: 'static {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>>;
}

pub struct Set(pub NodeId, pub f32);

impl<D: Device> GraphInstruction<D> for Set {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        graph.get_mut(self.0)?.dense_mut()?.set_to(self.1)?;

        Ok(())
    }
}
