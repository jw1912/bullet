use bullet_core::device::DeviceBuffer;

use crate::{backend::blas, Buffer, DenseMatrix};

pub fn linear_comb(
    ones: &Buffer<f32>,
    alpha: f32,
    input_a: &DenseMatrix,
    beta: f32,
    input_b: &DenseMatrix,
    output: &mut DenseMatrix,
) {
    assert_eq!(input_a.shape.without_batch_size(), input_b.shape.without_batch_size());

    match (input_a.shape.batch_size(), input_b.shape.batch_size()) {
        (Some(x), Some(y)) => {
            assert_eq!(x, y, "Batch sizes do not match: {x} != {y}");
            output.reshape_if_needed(input_a.shape);
            unsafe {
                blas::linear_comb_matrices(
                    output.buf.device().as_ref(),
                    output.shape.without_batch_size().size(),
                    x,
                    alpha,
                    Some(input_a.buf.ptr()),
                    beta,
                    input_b.buf.ptr(),
                    output.buf.mut_ptr(),
                );
            }
        }
        (None, Some(_)) => {
            copy_into_scaled(beta, input_b, output);
            add_assign_single_to_batched_scaled(ones, alpha, input_a, output);
        }
        (_, None) => {
            copy_into_scaled(alpha, input_a, output);
            add_assign_single_to_batched_scaled(ones, beta, input_b, output);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn linear_comb_backward(
    ones: &Buffer<f32>,
    alpha: f32,
    input_a: &DenseMatrix,
    input_a_grad: Option<&mut DenseMatrix>,
    beta: f32,
    input_b: &DenseMatrix,
    input_b_grad: Option<&mut DenseMatrix>,
    output_grad: &DenseMatrix,
) {
    if let Some(grd) = input_a_grad {
        backprop_add_single(ones, alpha, input_a, grd, output_grad);
    }

    if let Some(grd) = input_b_grad {
        backprop_add_single(ones, beta, input_b, grd, output_grad);
    }
}

fn copy_into_scaled(alpha: f32, input: &DenseMatrix, output: &mut DenseMatrix) {
    output.reshape_if_needed(input.shape);

    unsafe {
        blas::linear_comb_matrices(
            input.buf.device().as_ref(),
            output.shape.without_batch_size().size(),
            output.shape.batch_size().unwrap_or(1),
            alpha,
            Some(input.buf.ptr()),
            0.0,
            std::ptr::null(),
            output.buf.mut_ptr(),
        );
    }
}

fn add_assign_scaled(alpha: f32, input: &DenseMatrix, output: &mut DenseMatrix) {
    assert_eq!(input.shape, output.shape);

    unsafe {
        blas::linear_comb_matrices(
            input.buf.device().as_ref(),
            input.shape.without_batch_size().size(),
            input.shape.batch_size().unwrap_or(1),
            1.0,
            None,
            alpha,
            input.buf.ptr(),
            output.buf.mut_ptr(),
        );
    }
}

pub fn add_assign_single_to_batched_scaled(
    ones: &Buffer<f32>,
    alpha: f32,
    input: &DenseMatrix,
    output: &mut DenseMatrix,
) {
    assert_eq!(input.shape.batch_size().unwrap_or(1), 1);
    assert_eq!(input.shape.without_batch_size(), output.shape.without_batch_size());
    assert!(output.shape.batch_size().unwrap_or(1) <= ones.size());

    unsafe {
        blas::add_vector_to_matrix_columns(
            output.buf.device().as_ref(),
            output.shape.without_batch_size().size(),
            output.shape.batch_size().unwrap_or(1),
            alpha,
            ones.ptr(),
            input.buf.ptr(),
            output.buf.mut_ptr(),
        );
    }
}

pub fn backprop_add_single(
    ones: &Buffer<f32>,
    alpha: f32,
    input: &DenseMatrix,
    input_grad: &mut DenseMatrix,
    output_grad: &DenseMatrix,
) {
    assert_eq!(input.shape.without_batch_size(), output_grad.shape.without_batch_size());
    input_grad.reshape_if_needed(input.shape);

    assert!(output_grad.shape.cols() <= ones.size());

    match (input.shape.batch_size(), output_grad.shape.batch_size()) {
        (Some(_), Some(_)) | (None, None) => add_assign_scaled(alpha, output_grad, input_grad),
        (None, Some(x)) => unsafe {
            blas::reduce_add_cols(
                ones.device().as_ref(),
                output_grad.shape.without_batch_size().size(),
                x,
                ones.ptr(),
                output_grad.buf.ptr(),
                input_grad.buf.mut_ptr(),
                alpha,
                true,
            );
        },
        (Some(_), None) => unreachable!("Invalid shape pairs!"),
    }
}

pub fn reduce_add_batch(ones: &Buffer<f32>, input: &DenseMatrix, output: &mut DenseMatrix) {
    let shape = input.shape.without_batch_size();
    output.reshape_if_needed(shape);

    assert!(input.shape.batch_size().unwrap_or(1) <= ones.size());

    unsafe {
        blas::reduce_add_cols(
            input.buf.device().as_ref(),
            shape.size(),
            input.shape.batch_size().unwrap_or(1),
            ones.ptr(),
            input.buf.ptr(),
            output.buf.mut_ptr(),
            1.0,
            false,
        );
    }
}
