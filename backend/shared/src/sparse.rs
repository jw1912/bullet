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

use crate::{backend::ops, DenseMatrix, SparseMatrix};

pub fn copy_into_dense(sparse: &SparseMatrix, dense: &mut DenseMatrix) {
    dense.reshape_if_needed(sparse.shape);
    dense.set_zero();

    unsafe {
        ops::sparse_to_dense(
            sparse.shape.rows(),
            sparse.shape.batch_size().unwrap_or(1),
            sparse.nnz,
            sparse.buf.ptr(),
            dense.buf.mut_ptr(),
        );
    }
}
