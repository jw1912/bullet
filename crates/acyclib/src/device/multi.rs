use std::sync::Arc;

use crate::device::{
    Device,
    cpu::{CpuError, CpuThread},
    tensor::TensorRef,
};

pub trait MultiDeviceComm<D: Device> {
    fn new(devices: Vec<Arc<D>>) -> Self;

    fn reduce_sum_into_rank(&self, rank: usize, buffers: &[TensorRef<D>]) -> Result<(), D::DeviceError>;

    fn scatter_rank_into_rest(&self, rank: usize, buffers: &[TensorRef<D>]) -> Result<(), D::DeviceError>;
}

pub trait MultiDevice: Device {
    type Comm: MultiDeviceComm<Self>;
}

impl MultiDevice for CpuThread {
    type Comm = ();
}

impl MultiDeviceComm<CpuThread> for () {
    fn new(_: Vec<Arc<CpuThread>>) -> Self {}

    fn reduce_sum_into_rank(&self, rank: usize, buffers: &[TensorRef<CpuThread>]) -> Result<(), CpuError> {
        let mut buf = buffers[rank].dense_mut();

        for (i, other) in buffers.iter().enumerate() {
            if rank != i {
                buf.add(1.0, &other.dense()).map_err(|_| CpuError)?;
            }
        }

        Ok(())
    }

    fn scatter_rank_into_rest(&self, rank: usize, buffers: &[TensorRef<CpuThread>]) -> Result<(), CpuError> {
        let buf = buffers[rank].dense();

        for (i, other) in buffers.iter().enumerate() {
            if rank != i {
                other.dense_mut().copy_from(&buf)?;
            }
        }

        Ok(())
    }
}
