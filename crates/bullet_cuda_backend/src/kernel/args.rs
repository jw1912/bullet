use std::sync::Arc;

use acyclib::device::{
    OperationError,
    tensor::{Shape, TensorRef},
};
use cudarc::driver::CudaSlice;
use parking_lot::{MappedRwLockReadGuard, MappedRwLockWriteGuard};

use crate::{CudaDevice, CudaError, kernel::Expr};

#[derive(Clone, Debug)]
pub enum KernelInput {
    Size(Expr<i32>),
    F32(Expr<f32>),
    Slice { slice: TensorRef<CudaDevice>, layout: Option<usize>, mutable: bool, batched: bool, shape: Shape },
}

#[derive(Clone, Debug)]
pub struct KernelArgs {
    pub inputs: Vec<KernelInput>,
    pub grid_dim: [Expr<i32>; 3],
    pub block_dim: [Expr<i32>; 3],
    pub shared_mem_bytes: Expr<i32>,
}

impl KernelArgs {
    pub fn get_info(&self) -> Result<(Option<usize>, Arc<CudaDevice>), OperationError<CudaError>> {
        let mut batch_size = None;
        let mut device = None;

        for arg in &self.inputs {
            if let KernelInput::Slice { slice, batched, layout, shape, .. } = arg {
                let slice = slice.borrow();
                let this_batch_size = slice.batch_size();
                let this_device = slice.values.device();
                let this_ordinal = this_device.stream().context().ordinal();

                if this_batch_size.is_some() != *batched {
                    return Err(OperationError::MismatchedBatchSizes);
                }

                match layout {
                    None => {
                        if shape.size() != slice.dense()?.single_size() {
                            return Err(OperationError::InvalidTensorFormat);
                        }
                    }
                    Some(nnz) => {
                        if *nnz != slice.sparse()?.nnz() {
                            return Err(OperationError::InvalidTensorFormat);
                        }
                    }
                }

                match (batch_size, this_batch_size) {
                    (None, x) => batch_size = Some(x),
                    (Some(_), None) => {}
                    (Some(None), Some(x)) => batch_size = Some(Some(x)),
                    (Some(Some(x)), Some(y)) => {
                        if x != y {
                            return Err(OperationError::MismatchedBatchSizes);
                        }
                    }
                }

                if let Some((_, ordinal)) = device {
                    assert_eq!(ordinal, this_ordinal);
                }

                device = Some((this_device, this_ordinal));
            }
        }

        let batch_size = batch_size.ok_or(OperationError::UnsupportedOperation)?;
        let (device, _) = device.ok_or(OperationError::UnsupportedOperation)?;

        Ok((batch_size, device))
    }

    pub fn concretify<'a: 'b, 'b>(&'a self) -> Result<ConcreteKernelArgs<'b>, OperationError<CudaError>> {
        let (batch_size, device) = self.get_info()?;
        let batch_size = batch_size.unwrap_or(1) as i32;

        let grid_dim = self.grid_dim.iter().map(|x| x.evaluate(batch_size) as u32).collect::<Vec<_>>();
        let block_dim = self.block_dim.iter().map(|x| x.evaluate(batch_size) as u32).collect::<Vec<_>>();

        let concretify_input = |input: &'a KernelInput| match input {
            KernelInput::F32(x) => ConcreteKernelInput::F32(x.evaluate(batch_size as f32)),
            KernelInput::Size(x) => ConcreteKernelInput::Size(x.evaluate(batch_size)),
            KernelInput::Slice { slice, layout, mutable, .. } => match (layout, *mutable) {
                (None, false) => {
                    ConcreteKernelInput::SliceF32(MappedRwLockReadGuard::map(slice.dense(), |x| &x.buf.buf))
                }
                (Some(_), false) => {
                    ConcreteKernelInput::SliceI32(MappedRwLockReadGuard::map(slice.sparse(), |x| &x.buf.buf))
                }
                (None, true) => {
                    ConcreteKernelInput::MutSliceF32(MappedRwLockWriteGuard::map(slice.dense_mut(), |x| &mut x.buf.buf))
                }
                (Some(_), true) => {
                    ConcreteKernelInput::MutSliceI32(MappedRwLockWriteGuard::map(slice.sparse_mut(), |x| {
                        &mut x.buf.buf
                    }))
                }
            },
        };

        Ok(ConcreteKernelArgs {
            device,
            inputs: self.inputs.iter().map(concretify_input).collect(),
            grid_dim: (grid_dim[0], grid_dim[1], grid_dim[2]),
            block_dim: (block_dim[0], block_dim[1], block_dim[2]),
            shared_mem_bytes: self.shared_mem_bytes.evaluate(batch_size) as u32,
        })
    }
}

pub enum ConcreteKernelInput<'a> {
    Size(i32),
    F32(f32),
    SliceF32(MappedRwLockReadGuard<'a, CudaSlice<f32>>),
    SliceI32(MappedRwLockReadGuard<'a, CudaSlice<i32>>),
    MutSliceF32(MappedRwLockWriteGuard<'a, CudaSlice<f32>>),
    MutSliceI32(MappedRwLockWriteGuard<'a, CudaSlice<i32>>),
}

pub struct ConcreteKernelArgs<'a> {
    pub device: Arc<CudaDevice>,
    pub inputs: Vec<ConcreteKernelInput<'a>>,
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}
