use crate::{backend::ops, tensor::DenseMatrix};

use super::SparseMatrix;

impl SparseMatrix {
    pub fn softmax_across_columns_masked(input: &DenseMatrix, mask: &Self, output: &mut DenseMatrix) {
        assert_eq!(input.shape, mask.shape);
        output.reshape_if_needed(mask.shape);

        unsafe {
            ops::softmax_across_columns_masked(
                mask.max_active,
                mask.shape.rows(),
                mask.shape.cols(),
                mask.buf.ptr(),
                input.buf.ptr(),
                output.buf.mut_ptr(),
            );
        }
    }
}
