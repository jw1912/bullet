use std::sync::Arc;

use crate::device::{
    Device,
    cpu::{CpuError, CpuThread},
    tensor::TensorRef,
};

pub trait MultiDevice: Device {
    type Comm;

    fn make_comm(devices: Vec<Arc<Self>>) -> Self::Comm;

    fn reduce_sum_into_first(comm: &Self::Comm, buffers: &[TensorRef<Self>]) -> Result<(), Self::DeviceError>;

    fn scatter_first_into_rest(comm: &Self::Comm, buffers: &[TensorRef<Self>]) -> Result<(), Self::DeviceError>;
}

impl MultiDevice for CpuThread {
    type Comm = ();

    fn make_comm(_: Vec<Arc<Self>>) -> Self::Comm {}

    fn reduce_sum_into_first(_: &Self::Comm, buffers: &[TensorRef<Self>]) -> Result<(), Self::DeviceError> {
        let mut buf = buffers[0].dense_mut();

        for other in buffers.iter().skip(1) {
            buf.add(1.0, &other.dense()).map_err(|_| CpuError)?;
        }

        Ok(())
    }

    fn scatter_first_into_rest(_: &Self::Comm, buffers: &[TensorRef<Self>]) -> Result<(), Self::DeviceError> {
        let buf = buffers[0].dense();

        for other in buffers.iter().skip(1) {
            other.dense_mut().copy_from(&buf)?;
        }

        Ok(())
    }
}
