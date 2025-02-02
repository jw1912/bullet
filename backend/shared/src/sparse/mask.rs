use crate::{backend::ops, DenseMatrix, SparseMatrix};

pub fn mask(inputs: &DenseMatrix, masks: &SparseMatrix, outputs: &mut DenseMatrix) {
    let shape = inputs.shape;
    assert_eq!(shape, masks.shape);
    assert_eq!(shape.cols(), 1);
    assert!(masks.nnz <= shape.rows());

    outputs.reshape_if_needed(shape);
    outputs.set_zero();

    unsafe {
        ops::sparse_mask(
            shape.rows(),
            shape.batch_size().unwrap_or(1),
            masks.nnz,
            inputs.buf.ptr(),
            masks.buf.ptr(),
            outputs.buf.mut_ptr(),
        );
    }
}

pub fn backprop_mask(output_grads: &DenseMatrix, masks: &SparseMatrix, input_grads: &mut DenseMatrix) {
    let shape = output_grads.shape;
    assert_eq!(shape, masks.shape);
    assert_eq!(shape.cols(), 1);
    assert!(masks.nnz <= shape.rows());

    input_grads.reshape_if_needed(shape);

    unsafe {
        ops::sparse_mask_backprop(
            shape.rows(),
            shape.batch_size().unwrap_or(1),
            masks.nnz,
            output_grads.buf.ptr(),
            masks.buf.ptr(),
            input_grads.buf.mut_ptr(),
        );
    }
}
