mod buffer;
mod optimiser;
mod shape;
mod sparse;
mod tensor;
mod tensor_batch;

#[cfg(test)]
#[rustfmt::skip]
mod tests;

pub use buffer::GpuBuffer;
pub use bullet_cuda::{
    util::{device_synchronise, panic_if_cuda_error},
    CublasHandle,
};
pub use optimiser::Optimiser;
pub use shape::Shape;
pub use sparse::SparseTensor;
pub use tensor::Tensor;
pub use tensor_batch::TensorBatch;

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    ReLU,
    CReLU,
    SCReLU,
}
