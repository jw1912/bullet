mod backend;
pub mod dense;
//pub mod operations;
pub mod sparse;

use backend::util;
pub use dense::Activation;
pub use backend::{ExecutionContext, Buffer};

use bullet_core::{device::{Device, ValidType}, shape::Shape, tensor};

impl Device for ExecutionContext {
    type Buffer<T: ValidType> = Buffer<T>;
    type IdType = ();
    
    fn new(_: Self::IdType) -> Self {
        Self::default()    
    }

    fn synchronise(&self) {
        util::device_synchronise();
    }

    fn panic_if_device_error(&self) {
        util::panic_if_device_error("An error occurred on the device!");
    }
}

pub type DenseMatrix = tensor::DenseMatrix<ExecutionContext>;
pub type SparseMatrix = tensor::SparseMatrix<ExecutionContext>;
pub type Matrix = tensor::Matrix<ExecutionContext>;
pub type Tensor = tensor::Tensor<ExecutionContext>;
