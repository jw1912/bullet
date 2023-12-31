mod bindings;
mod buffer;
mod optimiser;
mod shape;
mod sparse;
mod tensor;
mod util;

#[cfg(test)]
#[rustfmt::skip]
mod tests;

pub use bindings::cublasHandle_t;
pub use buffer::GpuBuffer;
pub use optimiser::Optimiser;
pub use shape::Shape;
pub use sparse::SparseTensor;
pub use tensor::{Tensor, TensorBatch};
pub use util::{create_cublas_handle, device_synchronise, panic_if_cuda_error};

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    ReLU,
    CReLU,
    SCReLU,
}
