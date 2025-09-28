use std::sync::Arc;

use crate::{
    device::{
        Device, OperationError,
        multi::{MultiDevice, MultiDeviceComm},
        tensor::TensorRef,
    },
    graph::{Graph, GraphNodeId, like::GraphLike},
};

pub struct MultiDeviceGraph<D: Device + MultiDevice> {
    pub(super) comm: D::Comm,
    pub(super) graphs: Vec<Graph<D>>,
}

impl<D: Device + MultiDevice> MultiDeviceGraph<D> {
    pub fn get_weights(&self, id: &str) -> TensorRef<D> {
        self.primary().get_weights(id)
    }

    pub fn get_input(&self, id: &str) -> TensorRef<D> {
        self.primary().get_input(id)
    }
}

impl<D: Device + MultiDevice> GraphLike<D> for MultiDeviceGraph<D> {
    fn devices(&self) -> Vec<Arc<D>> {
        self.graphs.iter().map(Graph::device).collect()
    }

    fn primary(&self) -> &Graph<D> {
        &self.graphs[0]
    }

    fn primary_mut(&mut self) -> &mut Graph<D> {
        &mut self.graphs[0]
    }

    fn get_all(&self, id: GraphNodeId) -> Result<Vec<TensorRef<D>>, OperationError<D::DeviceError>> {
        self.graphs.iter().map(|g| g.get(id)).collect()
    }

    fn get_output_value(&self) -> Result<f32, OperationError<D::DeviceError>> {
        let mut sum = 0.0;

        for g in &self.graphs {
            sum += g.get_output_val()?;
        }

        Ok(sum)
    }

    fn execute_fn(&mut self, name: &str) -> Result<(), OperationError<D::DeviceError>> {
        self.comm.execute_fn(name, &mut self.graphs)?;
        Ok(())
    }

    fn reduce_sum_into_first(&self, buffers: &[TensorRef<D>]) -> Result<(), D::DeviceError> {
        self.comm.reduce_sum_into_rank(0, buffers)
    }

    fn scatter_first_into_rest(&self, buffers: &[TensorRef<D>]) -> Result<(), D::DeviceError> {
        self.comm.scatter_rank_into_rest(0, buffers)
    }
}
