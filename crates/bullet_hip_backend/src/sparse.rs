mod affine;
mod affine_dual;
mod gather;
mod mask;
mod select;
mod softmax;

pub use affine::*;
pub use affine_dual::*;
pub use gather::*;
pub use mask::*;
pub use select::*;
pub use softmax::*;

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
    unsafe {
        ops::sparse_to_dense(size, batch_size, nnz, sparse.ptr(), dense.mut_ptr());
    }

    Ok(())
}
