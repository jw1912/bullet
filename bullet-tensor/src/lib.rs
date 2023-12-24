mod bindings;
mod buffer;
mod optimiser;
mod shape;
mod sparse;
mod tensor;
mod util;

pub use buffer::GpuBuffer;
pub use optimiser::Optimiser;
pub use shape::Shape;
pub use sparse::SparseTensor;
pub use tensor::{Tensor, TensorBatch};
pub use util::{create_cublas_handle, panic_if_cuda_error, device_synchronise};

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    ReLU,
    CReLU,
    SCReLU,
}
