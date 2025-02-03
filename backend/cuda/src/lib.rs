mod blas;
mod buffer;
mod matmul;

use std::sync::Arc;

use buffer::CudaBuffer;
use bullet_core::{
    device::{Device, ValidType},
    tensor,
};

use cudarc::{
    cublas::CudaBlas,
    driver::CudaDevice,
};

pub type DenseMatrix = tensor::DenseMatrix<ExecutionContext>;
pub type SparseMatrix = tensor::SparseMatrix<ExecutionContext>;
pub type Matrix = tensor::Matrix<ExecutionContext>;
pub type Tensor = tensor::Tensor<ExecutionContext>;

pub struct ExecutionContext {
    device: Arc<CudaDevice>,
    blas: Arc<CudaBlas>,
}

impl Device for ExecutionContext {
    type Buffer<T: ValidType> = CudaBuffer<T>;
    type IdType = usize;

    fn new(id: Self::IdType) -> Self {
        let device = CudaDevice::new(id).unwrap();
        let blas = Arc::new(CudaBlas::new(device.clone()).unwrap());

        Self { device, blas }
    }

    fn synchronise(&self) {
        self.device.synchronize().unwrap();
    }

    // using `cudarc` we handle all errors at time of occurrence
    fn panic_if_device_error(&self, _: &str) {}
}
