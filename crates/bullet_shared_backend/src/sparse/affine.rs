use crate::{backend::ops, DenseMatrix, SparseMatrix};

pub fn affine(input_a: &DenseMatrix, input_b: &SparseMatrix, input_c: Option<&DenseMatrix>, output: &mut DenseMatrix) {
    assert!(input_a.shape.batch_size().is_none());
    assert_eq!(input_b.shape.cols(), 1);

    let output_shape = input_a.shape * input_b.shape;
    output.reshape_if_needed(output_shape);

    if let Some(c) = input_c {
        assert!(c.shape.batch_size().is_none());
        assert_eq!(c.shape.rows(), output_shape.rows());
        assert_eq!(c.shape.cols(), 1);
    }

    unsafe {
        ops::sparseAffineForward(
            input_b.shape.batch_size().unwrap_or(1),
            input_b.nnz,
            output.shape().rows(),
            input_a.buf.ptr(),
            input_c.map(|c| c.buf.ptr()).unwrap_or(std::ptr::null()),
            input_b.buf.ptr(),
            output.buf.mut_ptr(),
        );
    }
}

pub fn backprop_affine(
    input_a: &DenseMatrix,
    input_a_grad: &mut DenseMatrix,
    input_b: &SparseMatrix,
    input_c: Option<&DenseMatrix>,
    input_c_grad: Option<&mut DenseMatrix>,
    outputs: &DenseMatrix,
    output_grad: &DenseMatrix,
) {
    assert!(input_a.shape.batch_size().is_none());
    assert_eq!(input_b.shape.cols(), 1);
    assert_eq!(outputs.shape, output_grad.shape);

    let c_ptr = if let Some(grad) = input_c_grad {
        assert!(input_c.unwrap().shape.batch_size().is_none());
        grad.reshape_if_needed(input_c.unwrap().shape);
        grad.buf.mut_ptr()
    } else {
        std::ptr::null_mut()
    };

    input_a_grad.reshape_if_needed(input_a.shape());

    unsafe {
        ops::sparseAffineBackward(
            input_b.shape.batch_size().unwrap_or(1),
            input_b.nnz,
            output_grad.shape.rows(),
            input_a_grad.buf.mut_ptr(),
            c_ptr,
            input_b.buf.ptr(),
            outputs.buf.ptr(),
            output_grad.buf.ptr(),
        );
    }
}

pub fn linear(input_a: &DenseMatrix, input_b: &SparseMatrix, output: &mut DenseMatrix) {
    affine(input_a, input_b, None, output);
}

pub fn backprop_linear(
    input_a: &DenseMatrix,
    input_a_grad: &mut DenseMatrix,
    input_b: &SparseMatrix,
    outputs: &DenseMatrix,
    output_grad: &DenseMatrix,
) {
    backprop_affine(input_a, input_a_grad, input_b, None, None, outputs, output_grad);
}
