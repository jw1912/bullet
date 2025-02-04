use crate::{backend::ops, DenseMatrix, Shape, SparseMatrix};

pub fn gather(inputs: &DenseMatrix, indices: &SparseMatrix, outputs: &mut DenseMatrix) {
    assert!(indices.shape.batch_size().is_none());
    assert_eq!(indices.shape.cols(), 1);
    assert_eq!(indices.shape.rows(), indices.nnz);
    assert_eq!(inputs.shape.cols(), 1);

    let output_shape = Shape::from_raw(indices.shape.rows(), 1, inputs.shape.batch_size());
    outputs.reshape_if_needed(output_shape);
    outputs.set_zero();

    unsafe {
        ops::gather(
            inputs.shape.rows(),
            output_shape.rows(),
            output_shape.batch_size().unwrap_or(1),
            inputs.buf.ptr(),
            indices.buf.ptr(),
            outputs.buf.mut_ptr(),
        );
    }
}

pub fn backprop_gather(
    output_grads: &DenseMatrix,
    indices: &SparseMatrix,
    inputs: &DenseMatrix,
    input_grads: &mut DenseMatrix,
) {
    assert!(indices.shape.batch_size().is_none());
    assert_eq!(indices.shape.cols(), 1);
    assert_eq!(indices.shape.rows(), indices.nnz);

    assert_eq!(inputs.shape.cols(), 1);
    assert_eq!(output_grads.shape.cols(), 1);
    assert_eq!(output_grads.shape.batch_size(), inputs.shape.batch_size());
    assert_eq!(output_grads.shape.rows(), indices.shape.rows());

    input_grads.reshape_if_needed(inputs.shape);

    unsafe {
        ops::gather_backprop(
            inputs.shape.rows(),
            output_grads.shape.rows(),
            output_grads.shape.batch_size().unwrap_or(1),
            output_grads.buf.ptr(),
            indices.buf.ptr(),
            input_grads.buf.mut_ptr(),
        );
    }
}
