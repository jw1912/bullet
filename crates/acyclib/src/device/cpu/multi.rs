use std::sync::Arc;

use crate::{
    device::{
        OperationError,
        cpu::{CpuError, CpuThread},
        multi::{MultiDevice, MultiDeviceComm},
        tensor::TensorRef,
    },
    graph::Graph,
};

pub struct CpuComm;

impl MultiDevice for CpuThread {
    type Comm = CpuComm;
}

impl MultiDeviceComm<CpuThread> for CpuComm {
    fn new(_devices: Vec<Arc<CpuThread>>) -> Self {
        Self
    }

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

    fn execute_fn(&self, name: &str, graphs: &mut [Graph<CpuThread>]) -> Result<(), OperationError<CpuError>> {
        std::thread::scope(|scope| {
            graphs
                .iter_mut()
                .map(|graph| scope.spawn(|| graph.execute(name)))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Result<Vec<_>, _>>()
                .map(|_| ())
        })
    }
}
