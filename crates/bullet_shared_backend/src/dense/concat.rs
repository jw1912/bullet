use bullet_core::device::DeviceBuffer;

use crate::{backend::blas, DenseMatrix, Shape};

pub fn concat(input_a: &DenseMatrix, input_b: &DenseMatrix, output: &mut DenseMatrix) {
    assert_eq!(input_a.shape.cols(), 1);
    assert_eq!(input_b.shape.cols(), 1);
    assert_eq!(input_a.shape.batch_size(), input_b.shape.batch_size());

    let output_rows = input_a.shape.rows() + input_b.shape.rows();
    let output_shape = Shape::from_raw(output_rows, 1, input_a.shape.batch_size());
    output.reshape_if_needed(output_shape);

    unsafe {
        blas::copy_strided(
            input_a.buf.device().as_ref(),
            input_a.shape.rows(),
            input_a.shape.batch_size().unwrap_or(1),
            input_a.shape.rows(),
            input_a.buf.ptr(),
            output_shape.rows(),
            output.buf.mut_ptr(),
            false,
        );

        blas::copy_strided(
            input_b.buf.device().as_ref(),
            input_b.shape.rows(),
            input_b.shape.batch_size().unwrap_or(1),
            input_b.shape.rows(),
            input_b.buf.ptr(),
            output_shape.rows(),
            output.buf.mut_ptr().add(input_a.shape.rows()),
            false,
        );
    }
}

pub fn backprop_concat(
    input_a: &DenseMatrix,
    input_a_grad: Option<&mut DenseMatrix>,
    input_b: &DenseMatrix,
    input_b_grad: Option<&mut DenseMatrix>,
    output_grad: &DenseMatrix,
) {
    assert_eq!(input_a.shape.cols(), 1);
    assert_eq!(input_b.shape.cols(), 1);
    assert_eq!(output_grad.shape.cols(), 1);
    assert_eq!(input_a.shape.batch_size(), input_b.shape.batch_size());
    assert_eq!(input_a.shape.batch_size(), output_grad.shape.batch_size());

    if let Some(grad) = input_a_grad {
        grad.reshape_if_needed(input_a.shape);

        unsafe {
            blas::copy_strided(
                grad.buf.device().as_ref(),
                grad.shape.rows(),
                grad.shape.batch_size().unwrap_or(1),
                output_grad.shape.rows(),
                output_grad.buf.ptr(),
                grad.shape.rows(),
                grad.buf.mut_ptr(),
                true,
            );
        }
    }

    if let Some(grad) = input_b_grad {
        grad.reshape_if_needed(input_b.shape);

        unsafe {
            blas::copy_strided(
                grad.buf.device().as_ref(),
                grad.shape.rows(),
                grad.shape.batch_size().unwrap_or(1),
                output_grad.shape.rows(),
                output_grad.buf.ptr().add(input_a.shape.rows()),
                grad.shape.rows(),
                grad.buf.mut_ptr(),
                true,
            );
        }
    }
}
