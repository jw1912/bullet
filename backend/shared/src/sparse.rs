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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bullet_core::shape::Shape;

    use crate::{backend::util, ExecutionContext};

    use super::*;

    #[test]
    fn sparse_to_dense() {
        let device = Arc::new(ExecutionContext::default());

        let shape = Shape::new_batched(3, 1, 3);

        let mut input = SparseMatrix::zeroed(device.clone(), Shape::new(1, 1), 1);
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        util::panic_if_device_error("Failed to initialise matrices!");

        unsafe {
            input.load_from_slice(shape, 2, &[0, -1, 1, 2, 1, -1]);
        }

        copy_into_dense(&input, &mut output);

        let mut buf = [0.0; 9];
        output.write_to_slice(&mut buf);

        assert_eq!(buf, [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    }
}
