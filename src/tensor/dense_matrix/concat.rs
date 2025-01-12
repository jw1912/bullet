use crate::tensor::{
    backend::{ops, ExecutionContext},
    Shape,
};

use super::DenseMatrix;

impl DenseMatrix {
    pub fn concat(ctx: &mut ExecutionContext, input_a: &Self, input_b: &Self, output: &mut Self) {
        let cols = input_a.shape.cols();
        assert_eq!(cols, input_b.shape.cols());

        let output_shape = Shape::new(input_a.shape.rows() + input_b.shape.rows(), cols);
        output.reshape_if_needed(output_shape);

        unsafe {
            ops::copy_strided(
                ctx,
                input_a.shape.rows(),
                cols,
                input_a.shape.rows(),
                input_a.buf.ptr(),
                output_shape.rows(),
                output.buf.mut_ptr(),
                false,
            );

            ops::copy_strided(
                ctx,
                input_b.shape.rows(),
                cols,
                input_b.shape.rows(),
                input_b.buf.ptr(),
                output_shape.rows(),
                output.buf.mut_ptr().add(input_a.shape.rows()),
                false,
            );
        }
    }

    pub fn backprop_concat(
        ctx: &mut ExecutionContext,
        input_a: &Self,
        input_a_grad: Option<&mut Self>,
        input_b: &Self,
        input_b_grad: Option<&mut Self>,
        output_grad: &Self,
    ) {
        let cols = input_a.shape.cols();
        assert_eq!(cols, input_b.shape.cols());
        assert_eq!(cols, output_grad.shape.cols());

        if let Some(grad) = input_a_grad {
            grad.reshape_if_needed(input_a.shape);

            unsafe {
                ops::copy_strided(
                    ctx,
                    grad.shape.rows(),
                    grad.shape.cols(),
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
                ops::copy_strided(
                    ctx,
                    grad.shape.rows(),
                    grad.shape.cols(),
                    output_grad.shape.rows(),
                    output_grad.buf.ptr().add(input_a.shape.rows()),
                    grad.shape.rows(),
                    grad.buf.mut_ptr(),
                    true,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{backend::util, Shape};

    #[test]
    fn concat() {
        let mut ctx = ExecutionContext::default();

        let shape1 = Shape::new(3, 3);
        let shape2 = Shape::new(1, 3);

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

        // concat
        {
            DenseMatrix::concat(&mut ctx, &input1, &input2, &mut output);

            util::panic_if_device_error("Failed to concat matrices!");

            assert_eq!(output.shape(), Shape::new(4, 3));

            let mut buf = [0.0; 12];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, [-1.0, 4.0, 2.0, 1.0, -2.0, 0.0, -3.0, 2.0, 1.0, 1.0, 1.0, 3.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // de-concat
        {
            input1_grad.load_from_slice(shape1, &[1.0; 9]);

            DenseMatrix::backprop_concat(
                &mut ctx,
                &input1,
                Some(&mut input1_grad),
                &input2,
                Some(&mut input2_grad),
                &output,
            );

            util::panic_if_device_error("Failed to de-concat!");

            assert_eq!(input1_grad.shape(), shape1);
            assert_eq!(input2_grad.shape(), shape2);

            let mut grad1 = [0.0; 9];
            input1_grad.write_to_slice(&mut grad1);
            assert_eq!(grad1, [0.0, 5.0, 3.0, -1.0, 1.0, -2.0, 2.0, 2.0, 2.0]);

            let mut grad2 = [0.0; 3];
            input2_grad.write_to_slice(&mut grad2);
            assert_eq!(grad2, [1.0, 2.0, 3.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }
    }
}
