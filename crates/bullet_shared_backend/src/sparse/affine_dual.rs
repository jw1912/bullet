use crate::{backend::ops, Activation, DenseMatrix, SparseMatrix};
use bullet_core::shape::Shape;

pub fn affine_dual(
    input_a: &DenseMatrix,
    input_b1: &SparseMatrix,
    input_b2: &SparseMatrix,
    input_c: &DenseMatrix,
    output: &mut DenseMatrix,
    activation: Activation,
) {
    assert!(input_a.shape.batch_size().is_none());
    assert!(input_c.shape.batch_size().is_none());

    assert_eq!(input_b1.shape, input_b2.shape);
    assert_eq!(input_b1.shape.cols(), 1);
    assert_eq!(input_b1.nnz, input_b2.nnz);

    assert_eq!(input_c.shape.rows(), input_a.shape.rows());
    assert_eq!(input_c.shape.cols(), 1);

    let mut output_shape = input_a.shape * input_b1.shape;
    output_shape = Shape::from_raw(output_shape.rows() * 2, output_shape.cols(), output_shape.batch_size());

    output.reshape_if_needed(output_shape);

    unsafe {
        ops::sparseAffineDualForward(
            input_b1.shape.batch_size().unwrap_or(1),
            input_b1.nnz,
            input_a.shape().rows(),
            input_a.buf.ptr(),
            input_c.buf.ptr(),
            input_b1.buf.ptr(),
            input_b2.buf.ptr(),
            output.buf.mut_ptr(),
            activation as i32,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn backprop_affine_dual(
    input_a: &DenseMatrix,
    input_a_grad: &mut DenseMatrix,
    input_b1: &SparseMatrix,
    input_b2: &SparseMatrix,
    input_c: &DenseMatrix,
    input_c_grad: &mut DenseMatrix,
    outputs: &DenseMatrix,
    output_grad: &DenseMatrix,
    activation: Activation,
) {
    assert!(input_a.shape.batch_size().is_none());
    assert!(input_c.shape.batch_size().is_none());

    assert_eq!(input_b1.shape, input_b2.shape);
    assert_eq!(input_b1.shape.cols(), 1);
    assert_eq!(input_b1.nnz, input_b2.nnz);
    assert_eq!(outputs.shape, output_grad.shape);

    input_a_grad.reshape_if_needed(input_a.shape());
    input_c_grad.reshape_if_needed(input_c.shape());

    unsafe {
        ops::sparseAffineDualBackward(
            input_b1.shape.batch_size().unwrap_or(1),
            input_b1.nnz,
            input_a.shape.rows(),
            input_a_grad.buf.mut_ptr(),
            input_c_grad.buf.mut_ptr(),
            input_b1.buf.ptr(),
            input_b2.buf.ptr(),
            outputs.buf.ptr(),
            output_grad.buf.ptr(),
            activation as i32,
        );
    }
}
