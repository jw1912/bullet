use crate::{backend::ops, DenseMatrix, Shape, SparseMatrix};

fn softmax_across_columns_masked(mask: &SparseMatrix, input: &DenseMatrix, output: &mut DenseMatrix) {
    assert_eq!(input.shape, mask.shape);
    output.reshape_if_needed(Shape::new(mask.nnz, mask.shape.cols()));

    unsafe {
        ops::softmax_across_columns_masked(
            mask.nnz,
            mask.shape.rows(),
            mask.shape.cols(),
            mask.buf.ptr(),
            input.buf.ptr(),
            output.buf.mut_ptr(),
        );
    }
}

fn crossentropy_masked(
    mask: &SparseMatrix,
    pred: &DenseMatrix,
    target: &DenseMatrix,
    output: &mut DenseMatrix,
    error: &mut DenseMatrix,
) {
    assert_eq!(pred.shape, target.shape);
    assert_eq!(mask.shape.cols(), pred.shape.cols());
    assert_eq!(mask.nnz, pred.shape.rows());

    output.reshape_if_needed(pred.shape);
    error.reshape_if_needed(Shape::new(1, 1));
    error.set_zero();

    unsafe {
        ops::crossentropy_masked(
            mask.nnz,
            mask.shape.cols(),
            mask.buf.ptr(),
            pred.buf.ptr(),
            target.buf.ptr(),
            output.buf.mut_ptr(),
            error.buf.mut_ptr(),
        );
    }
}

pub fn softmax_crossentropy_loss_masked(
    mask: &SparseMatrix,
    input: &DenseMatrix,
    target: &DenseMatrix,
    output: &mut DenseMatrix,
    softmaxed: &mut DenseMatrix,
    individual_losses: &mut DenseMatrix,
) {
    assert_eq!(mask.shape, input.shape);
    assert_eq!(mask.shape.cols(), target.shape().cols());
    assert_eq!(mask.nnz, target.shape().rows());

    softmax_across_columns_masked(mask, input, softmaxed);

    crossentropy_masked(mask, softmaxed, target, individual_losses, output);
}

pub fn backprop_softmax_crossentropy_loss_masked(
    mask: &SparseMatrix,
    softmaxed: &DenseMatrix,
    target: &DenseMatrix,
    output_grad: &DenseMatrix,
    input_grad: &mut DenseMatrix,
) {
    assert_eq!(mask.shape.cols(), target.shape().cols());
    assert_eq!(mask.nnz, target.shape().rows());
    assert_eq!(softmaxed.shape(), target.shape());
    assert_eq!(output_grad.shape, Shape::new(1, 1));

    input_grad.reshape_if_needed(mask.shape);

    unsafe {
        ops::backprop_softmax_cross_entropy_masked(
            mask.nnz,
            mask.shape.rows(),
            mask.shape.cols(),
            mask.buf.ptr(),
            softmaxed.buf.ptr(),
            target.buf.ptr(),
            output_grad.buf.ptr(),
            input_grad.buf.mut_ptr(),
        );
    }
}
