use crate::tensor::backend::{ops, ExecutionContext};

use super::DenseMatrix;

impl DenseMatrix {
    pub fn linear_comb(
        ctx: &mut ExecutionContext,
        alpha: f32,
        input_a: &Self,
        beta: f32,
        input_b: &Self,
        output: &mut Self,
    ) {
        assert_eq!(input_a.shape.rows(), input_b.shape.rows());

        if input_a.shape == input_b.shape {
            output.reshape_if_needed(input_a.shape);
            unsafe {
                ops::linear_comb_matrices(
                    ctx,
                    output.shape.rows(),
                    output.shape.cols(),
                    alpha,
                    Some(input_a.buf.ptr()),
                    beta,
                    input_b.buf.ptr(),
                    output.buf.mut_ptr(),
                );
            }
        } else if input_a.shape.cols() == 1 {
            Self::copy_into_scaled(ctx, beta, input_b, output);
            Self::add_assign_vector_to_matrix_columns_scaled(ctx, alpha, input_a, output);
        } else if input_b.shape.cols() == 1 {
            Self::copy_into_scaled(ctx, alpha, input_a, output);
            Self::add_assign_vector_to_matrix_columns_scaled(ctx, beta, input_b, output);
        } else {
            panic!("Invalid shape pairs!")
        };
    }

    #[allow(clippy::too_many_arguments)]
    pub fn linear_comb_backward(
        ctx: &mut ExecutionContext,
        alpha: f32,
        input_a: &Self,
        input_a_grad: Option<&mut Self>,
        beta: f32,
        input_b: &Self,
        input_b_grad: Option<&mut Self>,
        output_grad: &Self,
    ) {
        if let Some(grd) = input_a_grad {
            Self::backprop_add_single(ctx, alpha, input_a, grd, output_grad);
        }

        if let Some(grd) = input_b_grad {
            Self::backprop_add_single(ctx, beta, input_b, grd, output_grad);
        }
    }

    fn copy_into_scaled(ctx: &mut ExecutionContext, alpha: f32, input: &Self, output: &mut Self) {
        output.reshape_if_needed(input.shape);

        unsafe {
            ops::linear_comb_matrices(
                ctx,
                output.shape.rows(),
                output.shape.cols(),
                alpha,
                Some(input.buf.ptr()),
                0.0,
                std::ptr::null(),
                output.buf.mut_ptr(),
            );
        }
    }

    pub fn add_assign_scaled(ctx: &mut ExecutionContext, alpha: f32, input: &Self, output: &mut Self) {
        assert_eq!(input.shape, output.shape);

        unsafe {
            ops::linear_comb_matrices(
                ctx,
                input.shape.rows(),
                input.shape.cols(),
                1.0,
                None,
                alpha,
                input.buf.ptr(),
                output.buf.mut_ptr(),
            );
        }
    }

    pub fn add_assign_vector_to_matrix_columns_scaled(
        ctx: &mut ExecutionContext,
        alpha: f32,
        input: &Self,
        output: &mut Self,
    ) {
        assert_eq!(input.shape.cols(), 1);
        assert_eq!(input.shape.rows(), output.shape.rows());

        unsafe {
            ops::add_vector_to_matrix_columns(
                ctx,
                output.shape.rows(),
                output.shape.cols(),
                alpha,
                input.buf.ptr(),
                output.buf.mut_ptr(),
            );
        }
    }

    pub fn backprop_add_single(
        ctx: &mut ExecutionContext,
        alpha: f32,
        input: &DenseMatrix,
        input_grad: &mut DenseMatrix,
        output_grad: &DenseMatrix,
    ) {
        input_grad.reshape_if_needed(input.shape);
        if input.shape.cols() == output_grad.shape.cols() {
            Self::add_assign_scaled(ctx, alpha, output_grad, input_grad);
        } else if input.shape.cols() == 1 {
            unsafe {
                ops::reduce_add_cols(
                    ctx,
                    output_grad.shape.rows(),
                    output_grad.shape.cols(),
                    output_grad.buf.ptr(),
                    input_grad.buf.mut_ptr(),
                    alpha,
                    true,
                );
            }
        } else {
            panic!("Invalid shape pairs!")
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{backend::util, Shape};

    #[test]
    fn linear_comb() {
        let mut ctx = ExecutionContext::default();

        let alpha = 2.0;
        let beta = -1.0;

        let shape1 = Shape::new(3, 3);
        let shape2 = Shape::new(3, 1);

        let mut input1 = DenseMatrix::default();
        let mut input2 = DenseMatrix::default();
        let mut input1_grad = DenseMatrix::default();
        let mut input2_grad = DenseMatrix::default();
        let mut output = DenseMatrix::default();

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
            DenseMatrix::linear_comb(&mut ctx, alpha, i, beta, j, &mut output);

            util::panic_if_device_error("Failed to add matrices!");

            assert_eq!(output.shape(), Shape::new(3, 3));

            let mut buf = [0.0; 9];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, expected_fwd[num], "{num}");

            util::panic_if_device_error("Failed to write data to CPU!");

            let (igrad, jgrad) =
                if num == 0 { (&mut input1_grad, &mut input2_grad) } else { (&mut input2_grad, &mut input1_grad) };

            igrad.set_zero();
            jgrad.set_zero();

            DenseMatrix::linear_comb_backward(&mut ctx, alpha, i, Some(igrad), beta, j, Some(jgrad), &output);

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

        DenseMatrix::linear_comb(&mut ctx, alpha, &input1, beta, &input1, &mut output);

        util::panic_if_device_error("Failed to add matrices!");

        assert_eq!(output.shape(), Shape::new(3, 3));

        let mut buf = [0.0; 9];
        output.write_to_slice(&mut buf);
        assert_eq!(buf, [-1.0, 4.0, 2.0, -2.0, 0.0, -3.0, 1.0, 1.0, 1.0]);

        util::panic_if_device_error("Failed to write data to CPU!");

        input1_grad.set_zero();
        input2_grad.set_zero();

        DenseMatrix::linear_comb_backward(
            &mut ctx,
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
}
