mod affine;
mod gather;
mod select;

pub use affine::*;
pub use gather::*;
pub use select::*;

use bullet_core::device::{DeviceBuffer, OperationError};

use crate::{
    backend::{ops, Buffer},
    OperationResult,
};

pub fn sparse_to_dense(
    batch_size: usize,
    size: usize,
    nnz: usize,
    sparse: &Buffer<i32>,
    dense: &mut Buffer<f32>,
) -> OperationResult {
    if batch_size * nnz > sparse.size() || batch_size * size > dense.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    dense.set_zero()?;

    unsafe {
        ops::sparse_to_dense(size, batch_size, nnz, sparse.ptr(), dense.mut_ptr());
    }

    Ok(())
}
