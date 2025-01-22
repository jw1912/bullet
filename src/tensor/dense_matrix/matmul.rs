use crate::tensor::backend::{blas, ExecutionContext};

use super::DenseMatrix;

impl DenseMatrix {
    pub(super) fn sgemm(
        ctx: &mut ExecutionContext,
        input_a: &Self,
        trans_a: bool,
        input_b: &Self,
        trans_b: bool,
        output: &mut Self,
        increment: bool,
    ) {
        let output_shape = input_a.shape.maybe_transpose(trans_a) * input_b.shape.maybe_transpose(trans_b);
        output.reshape_if_needed(output_shape);

        unsafe {
            blas::sgemm(
                ctx,
                input_a.buf.ptr(),
                input_a.shape.rows(),
                input_a.shape.cols(),
                trans_a,
                input_b.buf.ptr(),
                input_b.shape.rows(),
                input_b.shape.cols(),
                trans_b,
                output.buf.mut_ptr(),
                output.shape.rows(),
                output.shape.cols(),
                increment,
            );
        }
    }

    pub fn matmul(
        ctx: &mut ExecutionContext,
        input_a: &Self,
        trans_a: bool,
        input_b: &Self,
        trans_b: bool,
        output: &mut Self,
    ) {
        Self::sgemm(ctx, input_a, trans_a, input_b, trans_b, output, false);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backprop_matmul(
        ctx: &mut ExecutionContext,
        input_a: &Self,
        input_a_grad: Option<&mut Self>,
        trans_a: bool,
        input_b: &Self,
        input_b_grad: Option<&mut Self>,
        trans_b: bool,
        output_grad: &Self,
    ) {
        if let Some(grad_a) = input_a_grad {
            if trans_a {
                Self::sgemm(ctx, input_b, trans_b, output_grad, true, grad_a, true);
            } else {
                Self::sgemm(ctx, output_grad, false, input_b, !trans_b, grad_a, true);
            }
        }

        if let Some(grad_b) = input_b_grad {
            if trans_b {
                Self::sgemm(ctx, output_grad, true, input_a, trans_a, grad_b, true);
            } else {
                Self::sgemm(ctx, input_a, !trans_a, output_grad, false, grad_b, true);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{backend::util, Shape};

    #[test]
    fn matmul() {
        let mut ctx = ExecutionContext::default();

        let shape1 = Shape::new(2, 3);
        let shape2 = Shape::new(3, 1);

        let mut input1 = DenseMatrix::default();
        let mut input2 = DenseMatrix::default();
        let mut input1_grad = DenseMatrix::default();
        let mut input2_grad = DenseMatrix::default();
        let mut output = DenseMatrix::default();

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrices from CPU
        {
            input1.load_from_slice(
                shape1,
                // [ -1.0,  2.0,  0.0 ]
                // [  4.0, -2.0, -3.0 ]
                &[-1.0, 4.0, 2.0, -2.0, 0.0, -3.0],
            );

            input2.load_from_slice(
                shape2,
                // [ 1.0 ]
                // [ 2.0 ]
                // [ 3.0 ]
                &[1.0, 2.0, 3.0],
            );

            assert_eq!(input1.shape(), shape1);
            assert_eq!(input2.shape(), shape2);

            util::panic_if_device_error("Failed to load data from CPU!");
        }

        // normal matmul
        {
            DenseMatrix::matmul(&mut ctx, &input1, false, &input2, false, &mut output);

            util::panic_if_device_error("Failed to calculate matmul!");

            assert_eq!(output.shape(), Shape::new(2, 1));

            let mut buf = [0.0; 2];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, [3.0, -9.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // backprop normal matmul
        {
            DenseMatrix::backprop_matmul(
                &mut ctx,
                &input1,
                Some(&mut input1_grad),
                false,
                &input2,
                Some(&mut input2_grad),
                false,
                &output,
            );

            util::panic_if_device_error("Failed to backprop matmul!");

            assert_eq!(input1_grad.shape(), shape1);
            assert_eq!(input2_grad.shape(), shape2);

            let mut grad1 = [0.0; 6];
            input1_grad.write_to_slice(&mut grad1);
            assert_eq!(grad1, [3.0, -9.0, 6.0, -18.0, 9.0, -27.0]);

            let mut grad2 = [0.0; 3];
            input2_grad.write_to_slice(&mut grad2);
            assert_eq!(grad2, [-39.0, 24.0, 27.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // transposed matmul
        {
            DenseMatrix::matmul(&mut ctx, &input2, true, &input1, true, &mut output);

            util::panic_if_device_error("Failed to calculate transposed matmul!");

            assert_eq!(output.shape(), Shape::new(1, 2));

            let mut buf = [0.0; 2];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, [3.0, -9.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // backprop transposed matmul
        {
            input1_grad.set_zero();
            input2_grad.set_zero();

            DenseMatrix::backprop_matmul(
                &mut ctx,
                &input2,
                Some(&mut input2_grad),
                true,
                &input1,
                Some(&mut input1_grad),
                true,
                &output,
            );

            util::panic_if_device_error("Failed to backprop transposed matmul!");

            assert_eq!(input1_grad.shape(), shape1);
            assert_eq!(input2_grad.shape(), shape2);

            let mut grad1 = [0.0; 6];
            input1_grad.write_to_slice(&mut grad1);
            assert_eq!(grad1, [3.0, -9.0, 6.0, -18.0, 9.0, -27.0]);

            let mut grad2 = [0.0; 3];
            input2_grad.write_to_slice(&mut grad2);
            assert_eq!(grad2, [-39.0, 24.0, 27.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }
    }
}
