use std::sync::Arc;

use acyclib::device::{multi::MultiDevice, tensor::TensorRef};
use cudarc::nccl::{Comm, ReduceOp, group_end, group_start};

use crate::CudaDevice;

impl MultiDevice for CudaDevice {
    type Comm = Vec<Comm>;

    fn make_comm(devices: Vec<Arc<Self>>) -> Self::Comm {
        Comm::from_devices(devices.iter().map(|d| d.stream()).collect()).unwrap()
    }

    fn reduce_sum_into_first(comms: &Self::Comm, buffers: &[TensorRef<Self>]) -> Result<(), Self::DeviceError> {
        group_start().unwrap();

        for (buf, comm) in buffers.iter().zip(comms.iter()) {
            comm.reduce_in_place(&mut buf.dense_mut().buf.buf, &ReduceOp::Sum, 0).unwrap();
        }

        group_end().unwrap();

        Ok(())
    }

    fn scatter_first_into_rest(comms: &Self::Comm, buffers: &[TensorRef<Self>]) -> Result<(), Self::DeviceError> {
        group_start().unwrap();

        for (buf, comm) in buffers.iter().zip(comms.iter()) {
            comm.broadcast_in_place(&mut buf.dense_mut().buf.buf, 0).unwrap();
        }

        group_end().unwrap();

        Ok(())
    }
}
