mod blas;
mod buffer;
mod matmul;

use std::sync::Arc;

use buffer::CudaBuffer;
use bullet_core::{
    device::{Device, ValidType},
    graph::operation::Activation,
    shape::Shape,
    tensor,
};

use cudarc::{cublas::CudaBlas, driver::CudaDevice};

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

    fn activate(_input: &DenseMatrix, _output: &mut DenseMatrix, _activation: Activation) {
        unimplemented!()
    }

    fn sgemm(
        input_a: &DenseMatrix,
        shape_a: Shape,
        trans_a: bool,
        input_b: &DenseMatrix,
        shape_b: Shape,
        trans_b: bool,
        output: &mut DenseMatrix,
        output_shape: Shape,
        increment: bool,
    ) {
        matmul::sgemm(input_a, shape_a, trans_a, input_b, shape_b, trans_b, output, output_shape, increment);
    }

    fn sgemm_batched(
        input_a: &DenseMatrix,
        trans_a: bool,
        input_b: &DenseMatrix,
        trans_b: bool,
        output: &mut DenseMatrix,
        increment: bool,
    ) {
        matmul::sgemm_batched(input_a, trans_a, input_b, trans_b, output, increment);
    }
}
