use crate::backend::{ops, ExecutionContext};

use super::DenseMatrix;

impl DenseMatrix {
    pub fn add(ctx: &mut ExecutionContext, input_a: &Self, input_b: &Self, output: &mut Self) {
        assert_eq!(input_a.shape.rows(), input_b.shape.rows());

        if input_a.shape == input_b.shape {
            output.reshape_if_needed(input_a.shape);
            unsafe {
                ops::linear_comb_matrices(
                    ctx,
                    output.shape.rows(),
                    output.shape.cols(),
                    1.0,
                    input_a.buf.ptr(),
                    1.0,
                    input_b.buf.ptr(),
                    output.buf.mut_ptr(),
                );
            }
        } else if input_a.shape.cols() == 1 {
            input_b.copy_into(output);
            Self::add_assign_vector_to_matrix_columns(ctx, input_a, output);
        } else if input_b.shape.cols() == 1 {
            input_a.copy_into(output);
            Self::add_assign_vector_to_matrix_columns(ctx, input_b, output);
        } else {
            panic!("Invalid shape pairs!")
        };
    }

    pub fn add_backward(
        ctx: &mut ExecutionContext,
        input_a: &Self,
        input_a_grad: Option<&mut Self>,
        input_b: &Self,
        input_b_grad: Option<&mut Self>,
        output_grad: &Self,
    ) {
        if let Some(grd) = input_a_grad {
            Self::backprop_add_single(ctx, input_a, grd, output_grad);
        }

        if let Some(grd) = input_b_grad {
            Self::backprop_add_single(ctx, input_b, grd, output_grad);
        }
    }

    pub fn add_assign(ctx: &mut ExecutionContext, input: &Self, output: &mut Self) {
        assert_eq!(input.shape, output.shape);

        unsafe {
            ops::add_matrix_to(ctx, input.shape.rows(), input.shape.cols(), input.buf.ptr(), output.buf.mut_ptr());
        }
    }

    pub fn add_assign_vector_to_matrix_columns(ctx: &mut ExecutionContext, input: &Self, output: &mut Self) {
        assert_eq!(input.shape.cols(), 1);
        assert_eq!(input.shape.rows(), output.shape.rows());

        unsafe {
            ops::add_vector_to_matrix_columns(
                ctx,
                output.shape.rows(),
                output.shape.cols(),
                input.buf.ptr(),
                output.buf.mut_ptr(),
            );
        }
    }

    pub fn backprop_add_single(
        ctx: &mut ExecutionContext,
        input: &DenseMatrix,
        input_grad: &mut DenseMatrix,
        output_grad: &DenseMatrix,
    ) {
        input_grad.reshape_if_needed(input.shape);
        if input.shape.cols() == output_grad.shape.cols() {
            Self::add_assign(ctx, output_grad, input_grad);
        } else if input.shape.cols() == 1 {
            unsafe {
                ops::reduce_add_cols(
                    ctx,
                    output_grad.shape.rows(),
                    output_grad.shape.cols(),
                    output_grad.buf.ptr(),
                    input_grad.buf.mut_ptr(),
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
    use crate::{backend::util, tensor::Shape};

    #[test]
    fn add() {
        let mut ctx = ExecutionContext::default();

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

        // add
        for (num, (i, j)) in [(&input1, &input2), (&input2, &input1)].iter().enumerate() {
            DenseMatrix::add(&mut ctx, i, j, &mut output);

            util::panic_if_device_error("Failed to add matrices!");

            assert_eq!(output.shape(), Shape::new(3, 3));

            let mut buf = [0.0; 9];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, [0.0, 6.0, 5.0, -1.0, 2.0, 0.0, 2.0, 3.0, 4.0,], "{num}");

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // backprop add
        {
            DenseMatrix::add_backward(
                &mut ctx,
                &input1,
                Some(&mut input1_grad),
                &input2,
                Some(&mut input2_grad),
                &output,
            );

            util::panic_if_device_error("Failed to backprop addition!");

            assert_eq!(input1_grad.shape(), shape1);
            assert_eq!(input2_grad.shape(), shape2);

            let mut grad1 = [0.0; 9];
            input1_grad.write_to_slice(&mut grad1);
            assert_eq!(grad1, [0.0, 6.0, 5.0, -1.0, 2.0, 0.0, 2.0, 3.0, 4.0,]);

            let mut grad2 = [0.0; 3];
            input2_grad.write_to_slice(&mut grad2);
            assert_eq!(grad2, [1.0, 11.0, 9.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }
    }
}
