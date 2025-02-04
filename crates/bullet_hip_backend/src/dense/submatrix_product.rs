use bullet_core::device::DeviceBuffer;

use crate::{backend::blas, DenseMatrix, Shape};

#[allow(clippy::too_many_arguments)]
pub fn batched_sgemm(
    input_a: &DenseMatrix,
    shape_a: Shape,
    trans_a: bool,
    input_b: &DenseMatrix,
    shape_b: Shape,
    trans_b: bool,
    output: &mut DenseMatrix,
    increment: bool,
) {
    assert_eq!(shape_a.size(), input_a.shape.size());
    assert_eq!(shape_b.size(), input_b.shape.size());
    assert_eq!(input_a.shape.batch_size(), input_b.shape.batch_size());
    assert_eq!(input_a.shape.batch_size(), shape_a.batch_size());
    assert_eq!(shape_a.batch_size(), shape_b.batch_size());

    let output_shape = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);
    let batch_size = shape_a.batch_size().unwrap_or(1);

    unsafe {
        blas::batched_sgemm(
            input_a.buf.device().as_ref(),
            batch_size,
            input_a.buf.ptr(),
            shape_a.rows(),
            shape_a.cols(),
            trans_a,
            input_b.buf.ptr(),
            shape_b.rows(),
            shape_b.cols(),
            trans_b,
            output.buf.mut_ptr(),
            output_shape.rows(),
            output_shape.cols(),
            increment,
        );
    }
}

pub fn submatrix_product(key_size: usize, input_a: &DenseMatrix, input_b: &DenseMatrix, output: &mut DenseMatrix) {
    assert_eq!(input_a.shape.cols(), 1);
    assert_eq!(input_b.shape.cols(), 1);
    assert_eq!(input_a.shape.rows() % key_size, 0);
    assert_eq!(input_b.shape.rows() % key_size, 0);

    let batch_size = input_a.shape.batch_size();
    assert_eq!(batch_size, input_b.shape.batch_size());

    let shape_a = Shape::from_raw(key_size, input_a.shape.rows() / key_size, batch_size);
    let shape_b = Shape::from_raw(key_size, input_b.shape.rows() / key_size, batch_size);

    let output_size = shape_a.cols() * shape_b.cols();
    let output_shape = Shape::from_raw(output_size, 1, batch_size);
    output.reshape_if_needed(output_shape);
    batched_sgemm(input_a, shape_a, true, input_b, shape_b, false, output, false);
}

pub fn backprop_submatrix_product(
    key_size: usize,
    input_a: &DenseMatrix,
    input_a_grad: Option<&mut DenseMatrix>,
    input_b: &DenseMatrix,
    input_b_grad: Option<&mut DenseMatrix>,
    output_grad: &DenseMatrix,
) {
    assert_eq!(input_a.shape.cols(), 1);
    assert_eq!(input_b.shape.cols(), 1);
    assert_eq!(input_a.shape.rows() % key_size, 0);
    assert_eq!(input_b.shape.rows() % key_size, 0);

    let batch_size = input_a.shape.batch_size();
    assert_eq!(batch_size, input_b.shape.batch_size());
    assert_eq!(batch_size, output_grad.shape.batch_size());

    let shape_a = Shape::from_raw(key_size, input_a.shape.rows() / key_size, batch_size);
    let shape_b = Shape::from_raw(key_size, input_b.shape.rows() / key_size, batch_size);
    let output_shape = shape_a.transpose() * shape_b;

    assert_eq!(output_grad.shape.rows(), output_shape.without_batch_size().size());

    if let Some(grad_a) = input_a_grad {
        grad_a.reshape_if_needed(input_a.shape);
        batched_sgemm(input_b, shape_b, false, output_grad, output_shape, true, grad_a, true);
    }

    if let Some(grad_b) = input_b_grad {
        grad_b.reshape_if_needed(input_b.shape);
        batched_sgemm(input_a, shape_a, false, output_grad, output_shape, false, grad_b, true);
    }
}
