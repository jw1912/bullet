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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{backend::util, ExecutionContext, Shape};

    #[test]
    fn test_linear_comb() {
        let device = Arc::new(ExecutionContext::default());

        let alpha = 2.0;
        let beta = -1.0;

        let shape1 = Shape::new_batched(3, 1, 3);
        let shape2 = Shape::new(3, 1);

        let mut ones = Buffer::new(device.clone(), shape1.batch_size().unwrap());
        ones.load_from_slice(&vec![1.0; shape1.batch_size().unwrap()]);

        let mut input1 = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input2 = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input1_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input2_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrices from CPU
        {
            input1.load_from_slice(shape1, &[-1.0, 4.0, 2.0, -2.0, 0.0, -3.0, 1.0, 1.0, 1.0]);

            input2.load_from_slice(shape2, &[1.0, 2.0, 3.0]);

            assert_eq!(input1.shape(), shape1);
            assert_eq!(input2.shape(), shape2);

            util::panic_if_device_error("Failed to load data from CPU!");
        }

        let expected_fwd =
            [[-3.0, 6.0, 1.0, -5.0, -2.0, -9.0, 1.0, 0.0, -1.0], [3.0, 0.0, 4.0, 4.0, 4.0, 9.0, 1.0, 3.0, 5.0]];

        let expected_bwd1 = [
            [-6.0, 12.0, 2.0, -10.0, -4.0, -18.0, 2.0, 0.0, -2.0],
            [-3.0, 0.0, -4.0, -4.0, -4.0, -9.0, -1.0, -3.0, -5.0],
        ];

        let expected_bwd2 = [[7.0, -4.0, 9.0], [16.0, 14.0, 36.0]];

        for (num, (i, j)) in [(&input1, &input2), (&input2, &input1)].iter().enumerate() {
            linear_comb(&ones, alpha, i, beta, j, &mut output);

            util::panic_if_device_error("Failed to add matrices!");

            assert_eq!(output.shape(), Shape::new_batched(3, 1, 3));

            let mut buf = [0.0; 9];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, expected_fwd[num], "{num}");

            util::panic_if_device_error("Failed to write data to CPU!");

            let (igrad, jgrad) =
                if num == 0 { (&mut input1_grad, &mut input2_grad) } else { (&mut input2_grad, &mut input1_grad) };

            igrad.set_zero();
            jgrad.set_zero();

            linear_comb_backward(&ones, alpha, i, Some(igrad), beta, j, Some(jgrad), &output);

            util::panic_if_device_error("Failed to backprop addition!");

            assert_eq!(input1_grad.shape(), shape1);
            assert_eq!(input2_grad.shape(), shape2);

            let mut grad1 = [0.0; 9];
            input1_grad.write_to_slice(&mut grad1);
            assert_eq!(grad1, expected_bwd1[num], "{num}");

            let mut grad2 = [0.0; 3];
            input2_grad.write_to_slice(&mut grad2);
            assert_eq!(grad2, expected_bwd2[num], "{num}");

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        linear_comb(&ones, alpha, &input1, beta, &input1, &mut output);

        util::panic_if_device_error("Failed to add matrices!");

        assert_eq!(output.shape(), Shape::new_batched(3, 1, 3));

        let mut buf = [0.0; 9];
        output.write_to_slice(&mut buf);
        assert_eq!(buf, [-1.0, 4.0, 2.0, -2.0, 0.0, -3.0, 1.0, 1.0, 1.0]);

        util::panic_if_device_error("Failed to write data to CPU!");

        input1_grad.set_zero();
        input2_grad.set_zero();

        linear_comb_backward(
            &ones,
            alpha,
            &input1,
            Some(&mut input1_grad),
            beta,
            &input1,
            Some(&mut input2_grad),
            &output,
        );

        util::panic_if_device_error("Failed to backprop addition!");

        assert_eq!(input1_grad.shape(), shape1);
        assert_eq!(input2_grad.shape(), shape1);

        let mut grad1 = [0.0; 9];
        input1_grad.write_to_slice(&mut grad1);
        assert_eq!(grad1, [-2.0, 8.0, 4.0, -4.0, 0.0, -6.0, 2.0, 2.0, 2.0]);

        let mut grad2 = [0.0; 9];
        input2_grad.write_to_slice(&mut grad2);
        assert_eq!(grad2, [1.0, -4.0, -2.0, 2.0, 0.0, 3.0, -1.0, -1.0, -1.0]);

        util::panic_if_device_error("Failed to write data to CPU!");
    }

    #[test]
    fn test_reduce_add_cols() {
        let device = Arc::new(ExecutionContext::default());

        let mut ones = Buffer::new(device.clone(), 9);
        ones.load_from_slice(&[1.0; 9]);

        let mut input = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        input.load_from_slice(Shape::new_batched(1, 1, 9), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        reduce_add_batch(&ones, &input, &mut output);

        assert_eq!(output.shape, Shape::new(1, 1));
        let mut buf = [0.0];
        output.write_to_slice(&mut buf);
        assert_eq!(buf, [45.0]);
    }
}
