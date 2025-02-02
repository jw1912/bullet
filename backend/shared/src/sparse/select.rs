use crate::{backend::ops, DenseMatrix, Shape, SparseMatrix};

pub fn select(input: &DenseMatrix, indices: &SparseMatrix, output: &mut DenseMatrix) {
    let rows = input.shape.rows();
    let buckets = indices.shape.rows();

    assert_eq!(input.shape.batch_size(), indices.shape.batch_size());
    assert_eq!(input.shape.cols(), 1);
    assert_eq!(indices.shape.cols(), 1);
    assert_eq!(indices.nnz, 1);

    assert_eq!(rows % buckets, 0, "Cannot divide vector evenly among buckets!");
    let output_rows = rows / buckets;
    let shape = Shape::from_raw(output_rows, 1, input.shape.batch_size());
    output.reshape_if_needed(shape);

    unsafe {
        ops::selectForward(
            input.shape.batch_size().unwrap_or(1),
            rows,
            output_rows,
            indices.buf.ptr(),
            input.buf.ptr(),
            output.buf.mut_ptr(),
        );
    }
}

pub fn select_backprop(
    input: &DenseMatrix,
    indices: &SparseMatrix,
    output_grad: &DenseMatrix,
    input_grad: &mut DenseMatrix,
) {
    let rows = input.shape.rows();
    let buckets = indices.shape.rows();

    assert_eq!(input.shape.batch_size(), indices.shape.batch_size());
    assert_eq!(input.shape.batch_size(), output_grad.shape.batch_size());
    assert_eq!(input.shape.cols(), 1);
    assert_eq!(output_grad.shape.cols(), 1);
    assert_eq!(indices.shape.cols(), 1);
    assert_eq!(indices.nnz, 1);

    assert_eq!(rows % buckets, 0, "Cannot divide vector evenly among buckets!");
    let output_rows = rows / buckets;
    assert_eq!(output_rows, output_grad.shape.rows());

    input_grad.reshape_if_needed(input.shape);

    unsafe {
        ops::selectBackprop(
            input.shape.batch_size().unwrap_or(1),
            rows,
            output_rows,
            indices.buf.ptr(),
            output_grad.buf.ptr(),
            input_grad.buf.mut_ptr(),
        );
    }
}
