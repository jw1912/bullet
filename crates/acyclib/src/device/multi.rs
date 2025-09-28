use std::sync::Arc;

use crate::{
    device::{Device, OperationError, tensor::TensorRef},
    graph::Graph,
};

pub trait MultiDeviceComm<D: Device> {
    fn new(devices: Vec<Arc<D>>) -> Self;

    fn reduce_sum_into_rank(&self, rank: usize, buffers: &[TensorRef<D>]) -> Result<(), D::DeviceError>;

    fn scatter_rank_into_rest(&self, rank: usize, buffers: &[TensorRef<D>]) -> Result<(), D::DeviceError>;

    fn execute_fn(&self, name: &str, graphs: &mut [Graph<D>]) -> Result<(), OperationError<D::DeviceError>>;
}

pub trait MultiDevice: Device {
    type Comm: MultiDeviceComm<Self>;
}
