use std::cell::{Ref, RefMut};

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
    pub inputs: Vec<KernelInput<D>>,
    pub grid_dim: [VariableExpression<i32>; 3],
    pub block_dim: [VariableExpression<i32>; 3],
    pub shared_mem_bytes: VariableExpression<i32>,
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

    pub fn concretify<'a: 'b, 'b>(&'a self) -> Result<ConcreteKernelArgs<'b, D>, OperationError<D::DeviceError>> {
        let batch_size = self.get_batch_size()?.unwrap_or(1) as i32;

        let grid_dim = self.grid_dim.iter().map(|x| x.evaluate(batch_size) as u32).collect::<Vec<_>>();
        let block_dim = self.block_dim.iter().map(|x| x.evaluate(batch_size) as u32).collect::<Vec<_>>();

        let concretify_input = |input: &'a KernelInput<D>| match input {
            KernelInput::F32(x) => ConcreteKernelInput::F32(x.evaluate(batch_size as f32)),
            KernelInput::Size(x) => ConcreteKernelInput::Size(x.evaluate(batch_size)),
            KernelInput::SliceF32(x) => ConcreteKernelInput::SliceF32(Ref::map(x.dense(), |x| &x.buf)),
            KernelInput::SliceI32(x) => ConcreteKernelInput::SliceI32(Ref::map(x.sparse(), |x| &x.buf)),
            KernelInput::MutSliceF32(x) => ConcreteKernelInput::MutSliceF32(RefMut::map(x.dense_mut(), |x| &mut x.buf)),
            KernelInput::MutSliceI32(x) => {
                ConcreteKernelInput::MutSliceI32(RefMut::map(x.sparse_mut(), |x| &mut x.buf))
            }
        };

        Ok(ConcreteKernelArgs {
            inputs: self.inputs.iter().map(concretify_input).collect(),
            grid_dim: (grid_dim[0], grid_dim[1], grid_dim[2]),
            block_dim: (block_dim[0], block_dim[1], block_dim[2]),
            shared_mem_bytes: self.shared_mem_bytes.evaluate(batch_size) as u32,
        })
    }
}

pub enum ConcreteKernelInput<'a, D: Device> {
    Size(i32),
    F32(f32),
    SliceF32(Ref<'a, D::BufferF32>),
    SliceI32(Ref<'a, D::BufferI32>),
    MutSliceF32(RefMut<'a, D::BufferF32>),
    MutSliceI32(RefMut<'a, D::BufferI32>),
}

pub struct ConcreteKernelArgs<'a, D: Device> {
    pub inputs: Vec<ConcreteKernelInput<'a, D>>,
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}
