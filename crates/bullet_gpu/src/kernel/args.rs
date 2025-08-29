use bullet_core::{
    device::{Device, OperationError},
    tensor::TensorRef,
};

use crate::kernel::expr::VariableExpression;

#[derive(Clone)]
pub enum KernelInput<D: Device> {
    Size(VariableExpression<i32>),
    F32(VariableExpression<f32>),
    SliceF32(TensorRef<D>),
    SliceI32(TensorRef<D>),
    MutSliceF32(TensorRef<D>),
    MutSliceI32(TensorRef<D>),
}

pub struct KernelArgs<D: Device> {
    inputs: Vec<KernelInput<D>>,
    grid_dim: [VariableExpression<i32>; 3],
    block_dim: [VariableExpression<i32>; 3],
    shared_mem_bytes: VariableExpression<i32>,
}

impl<D: Device> KernelArgs<D> {
    pub fn get_batch_size(&self) -> Result<Option<usize>, OperationError<D::DeviceError>> {
        let mut batch_size = None;

        for arg in &self.inputs {
            match arg {
                KernelInput::SliceF32(x)
                | KernelInput::SliceI32(x)
                | KernelInput::MutSliceF32(x)
                | KernelInput::MutSliceI32(x) => {
                    let this_batch_size = x.borrow().batch_size();

                    if let Some(size) = batch_size {
                        if size != this_batch_size {
                            return Err(OperationError::MismatchedBatchSizes);
                        }
                    } else {
                        batch_size = Some(this_batch_size);
                    }
                }
                _ => {}
            }
        }

        batch_size.ok_or(OperationError::UnsupportedOperation)
    }
}
