mod buffer;
mod optimiser;
mod shape;
mod sparse;
mod tensor;
mod tensor_batch;

#[cfg(test)]
#[rustfmt::skip]
mod tests;

use bullet_cuda as backend;

pub use backend::{util::{device_name, device_synchronise, panic_if_device_error}, DeviceHandles};
pub use buffer::DeviceBuffer;
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
