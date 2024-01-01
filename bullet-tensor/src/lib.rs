mod buffer;
mod optimiser;
mod shape;
mod sparse;
mod tensor;

#[cfg(test)]
#[rustfmt::skip]
mod tests;

pub use buffer::GpuBuffer;
pub use optimiser::Optimiser;
pub use shape::Shape;
pub use sparse::SparseTensor;
pub use tensor::{Tensor, TensorBatch};
pub use bullet_cuda::{CublasHandle, util::{device_synchronise, panic_if_cuda_error}};

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    ReLU,
    CReLU,
    SCReLU,
}
