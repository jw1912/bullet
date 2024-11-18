use crate::{
    backend::{ops, ExecutionContext},
    Shape,
};

use super::DenseMatrix;

impl DenseMatrix {
    pub fn slice_rows(ctx: &mut ExecutionContext, input: &Self, start: usize, end: usize, output: &mut Self) {
        assert!(end > start, "Invalid slice indices! end = {end} > start = {start}");

        let output_shape = Shape::new(end - start, input.shape.cols());
        output.reshape_if_needed(output_shape);

        unsafe {
            ops::copy_strided(
                ctx,
                output_shape.rows(),
                input.shape.cols(),
                input.shape.rows(),
                input.buf.ptr().add(start),
                output_shape.rows(),
                output.buf.mut_ptr(),
                false,
            );
        }
    }

    pub fn backprop_slice_rows(
        ctx: &mut ExecutionContext,
        input: &Self,
        input_grad: Option<&mut Self>,
        start: usize,
        end: usize,
        output_grad: &Self,
    ) {
        assert!(end > start, "Invalid slice indices! end = {end} > start = {start}");
        let output_shape = Shape::new(end - start, input.shape.cols());
        assert_eq!(output_shape, output_grad.shape);

        if let Some(grad) = input_grad {
            grad.reshape_if_needed(input.shape);

            unsafe {
                ops::copy_strided(
                    ctx,
                    output_shape.rows(),
                    output_shape.cols(),
                    output_shape.rows(),
                    output_grad.buf.ptr(),
                    grad.shape.rows(),
                    grad.buf.mut_ptr().add(start),
                    true,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::util, tensor::Shape};

    #[test]
    fn slice() {
        let mut ctx = ExecutionContext::default();

        let shape = Shape::new(3, 3);

        let mut input = DenseMatrix::default();
        let mut input_grad = DenseMatrix::default();
        let mut output = DenseMatrix::default();

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrices from CPU
        {
            input.load_from_slice(shape, &[-1.0, 4.0, 2.0, -2.0, 0.0, -3.0, 1.0, 1.0, 1.0]);

            assert_eq!(input.shape(), shape);

            util::panic_if_device_error("Failed to load data from CPU!");
        }

        // concat
        {
            DenseMatrix::slice_rows(&mut ctx, &input, 0, 2, &mut output);

            util::panic_if_device_error("Failed to concat matrices!");

            assert_eq!(output.shape(), Shape::new(2, 3));

            let mut buf = [0.0; 6];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, [-1.0, 4.0, -2.0, 0.0, 1.0, 1.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // de-concat
        {
            input_grad.load_from_slice(shape, &[1.0; 9]);

            DenseMatrix::backprop_slice_rows(&mut ctx, &input, Some(&mut input_grad), 0, 2, &output);

            util::panic_if_device_error("Failed to backprop slice!");

            assert_eq!(input_grad.shape(), shape);

            let mut grad = [0.0; 9];
            input_grad.write_to_slice(&mut grad);
            assert_eq!(grad, [0.0, 5.0, 1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 1.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }
    }
}
