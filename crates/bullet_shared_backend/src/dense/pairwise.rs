use crate::{backend::ops, DenseMatrix, Shape};

pub fn pairwise(input: &DenseMatrix, output: &mut DenseMatrix, post_concat: bool) {
    assert_eq!(input.shape.cols(), 1);
    let mut rows = input.shape.rows();
    let mut batch_size = input.shape.batch_size().unwrap_or(1);

    let shape = Shape::new_batched(rows / 2, 1, batch_size);

    if post_concat {
        rows /= 2;
        batch_size *= 2;
    }

    assert_eq!(rows % 2, 0);
    output.reshape_if_needed(shape);

    unsafe {
        ops::pairwiseMul(batch_size, rows / 2, input.buf.ptr(), output.buf.mut_ptr());
    }
}

pub fn backprop_pairwise(
    input: &DenseMatrix,
    output_grad: &DenseMatrix,
    input_grad: &mut DenseMatrix,
    post_concat: bool,
) {
    assert_eq!(input.shape.cols(), 1);
    let mut rows = input.shape.rows();
    let mut batch_size = input.shape.batch_size().unwrap_or(1);

    if post_concat {
        rows /= 2;
        batch_size *= 2;
    }

    input_grad.reshape_if_needed(input.shape);

    unsafe {
        ops::backpropPairwiseMul(
            batch_size,
            rows / 2,
            input.buf.ptr(),
            output_grad.buf.ptr(),
            input_grad.buf.mut_ptr(),
        );
    }
}
