use std::sync::Arc;

use crate::{
    device::{Device, OperationError, tensor::TensorRef},
    graph::{Graph, GraphNodeId},
};

pub trait GraphLike<D: Device> {
    fn devices(&self) -> Vec<Arc<D>>;

    fn primary(&self) -> &Graph<D>;

    fn primary_mut(&mut self) -> &mut Graph<D>;

    fn get_all(&self, id: GraphNodeId) -> Vec<Result<TensorRef<D>, OperationError<D::DeviceError>>>;

    fn get_output_value(&self) -> Result<f32, OperationError<D::DeviceError>>;

    fn execute_fn(&mut self, name: &str) -> Result<(), OperationError<D::DeviceError>>;
}

impl<D: Device> GraphLike<D> for Graph<D> {
    fn devices(&self) -> Vec<Arc<D>> {
        vec![self.device()]
    }

    fn primary(&self) -> &Graph<D> {
        self
    }

    fn primary_mut(&mut self) -> &mut Graph<D> {
        self
    }

    fn get_all(&self, id: GraphNodeId) -> Vec<Result<TensorRef<D>, OperationError<<D as Device>::DeviceError>>> {
        vec![self.get(id)]
    }

    fn get_output_value(&self) -> Result<f32, OperationError<<D as Device>::DeviceError>> {
        self.get_output_val()
    }

    fn execute_fn(&mut self, name: &str) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        self.execute(name)
    }
}
