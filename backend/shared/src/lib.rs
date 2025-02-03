mod backend;
pub mod dense;
mod matmul;
pub mod sparse;

use backend::{util, Buffer};
pub use backend::ExecutionContext;
pub use dense::Activation;

use bullet_core::{
    device::{Device, ValidType},
    shape::Shape,
    tensor,
};

pub type DenseMatrix = tensor::DenseMatrix<ExecutionContext>;
pub type SparseMatrix = tensor::SparseMatrix<ExecutionContext>;
pub type Matrix = tensor::Matrix<ExecutionContext>;
pub type Tensor = tensor::Tensor<ExecutionContext>;

impl Device for ExecutionContext {
    type Buffer<T: ValidType> = Buffer<T>;
    type IdType = ();

    fn new(_: Self::IdType) -> Self {
        Self::default()
    }

    fn synchronise(&self) {
        util::device_synchronise();
    }

    fn panic_if_device_error(&self, msg: &str) {
        util::panic_if_device_error(msg);
    }
}
