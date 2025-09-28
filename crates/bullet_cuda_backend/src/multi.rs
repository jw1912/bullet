use std::{collections::HashSet, sync::Arc};

use acyclib::{
    device::{
        OperationError,
        multi::{MultiDevice, MultiDeviceComm},
        tensor::TensorRef,
    },
    graph::Graph,
};
use cudarc::nccl::{Comm, ReduceOp, group_end, group_start};

use crate::{CudaDevice, CudaError};

pub struct CudaComm(Vec<Comm>);

impl MultiDeviceComm<CudaDevice> for CudaComm {
    fn new(devices: Vec<Arc<CudaDevice>>) -> Self {
        let ids = devices.iter().map(|d| d.stream().context().cu_device());
        if ids.collect::<HashSet<_>>().len() != devices.len() {
            panic!("Cannot use the same CUDA device more than once!");
        }

        CudaComm(Comm::from_devices(devices.iter().map(|d| d.stream()).collect()).unwrap())
    }

    fn reduce_sum_into_rank(&self, rank: usize, buffers: &[TensorRef<CudaDevice>]) -> Result<(), CudaError> {
        group_start().map_err(CudaError::Nccl)?;

        for (buf, comm) in buffers.iter().zip(self.0.iter()) {
            comm.reduce_in_place(&mut buf.dense_mut().buf.buf, &ReduceOp::Sum, rank as i32).map_err(CudaError::Nccl)?;
        }

        group_end().map_err(CudaError::Nccl)?;

        Ok(())
    }

    fn scatter_rank_into_rest(&self, rank: usize, buffers: &[TensorRef<CudaDevice>]) -> Result<(), CudaError> {
        group_start().map_err(CudaError::Nccl)?;

        for (buf, comm) in buffers.iter().zip(self.0.iter()) {
            comm.broadcast_in_place(&mut buf.dense_mut().buf.buf, rank as i32).map_err(CudaError::Nccl)?;
        }

        group_end().map_err(CudaError::Nccl)?;

        Ok(())
    }

    fn execute_fn(&self, name: &str, graphs: &mut [Graph<CudaDevice>]) -> Result<(), OperationError<CudaError>> {
        for graph in graphs {
            graph.execute(name)?;
        }

        Ok(())
    }
}

impl MultiDevice for CudaDevice {
    type Comm = CudaComm;
}
