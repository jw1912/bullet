use crate::{
    backend::{ops, ExecutionContext},
    Shape,
};

use super::DenseMatrix;

impl DenseMatrix {
    #[allow(clippy::too_many_arguments)]
    fn batched_sgemm(
        ctx: &mut ExecutionContext,
        input_a: &Self,
        shape_a: Shape,
        trans_a: bool,
        input_b: &Self,
        shape_b: Shape,
        trans_b: bool,
        output: &mut Self,
        increment: bool,
    ) {
        assert_eq!(shape_a.size(), input_a.shape.rows());
        assert_eq!(shape_b.size(), input_b.shape.rows());
        assert_eq!(input_a.shape.cols(), input_b.shape.cols());

        let output_shape = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);
        let batch_size = input_a.shape.cols();

        output.reshape_if_needed(Shape::new(output_shape.size(), batch_size));

        unsafe {
            ops::batched_sgemm(
                ctx,
                batch_size,
                input_a.buf.ptr(),
                shape_a.rows(),
                shape_a.cols(),
                trans_a,
                input_b.buf.ptr(),
                shape_b.rows(),
                shape_b.cols(),
                trans_b,
                output.buf.mut_ptr(),
                output_shape.rows(),
                output_shape.cols(),
                increment,
            );
        }
    }

    pub fn submatrix_product(
        ctx: &mut ExecutionContext,
        key_size: usize,
        input_a: &Self,
        input_b: &Self,
        output: &mut Self,
    ) {
        assert_eq!(input_a.shape, input_b.shape);
        assert_eq!(input_a.shape.rows() % key_size, 0);

        let shape = Shape::new(key_size, input_a.shape.rows() / key_size);

        Self::batched_sgemm(ctx, input_a, shape, true, input_b, shape, false, output, false);
    }

    pub fn backprop_submatrix_product(
        ctx: &mut ExecutionContext,
        key_size: usize,
        input_a: &Self,
        input_a_grad: Option<&mut Self>,
        input_b: &Self,
        input_b_grad: Option<&mut Self>,
        output_grad: &Self,
    ) {
        assert_eq!(input_a.shape, input_b.shape);
        assert_eq!(input_a.shape.rows() % key_size, 0);

        let shape = Shape::new(key_size, input_a.shape.rows() / key_size);
        let output_shape = shape.transpose() * shape;

        assert_eq!(output_grad.shape.rows(), output_shape.size());

        if let Some(grad_a) = input_a_grad {
            Self::batched_sgemm(ctx, input_b, shape, false, output_grad, output_shape, true, grad_a, true);
        }

        if let Some(grad_b) = input_b_grad {
            Self::batched_sgemm(ctx, input_a, shape, false, output_grad, output_shape, false, grad_b, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::util, tensor::Shape};

    #[test]
    fn submatrix_product() {
        let mut ctx = ExecutionContext::default();

        let shape = Shape::new(2, 3);
        let key_size = 1;

        let mut input1 = DenseMatrix::default();
        let mut input2 = DenseMatrix::default();
        let mut input1_grad = DenseMatrix::default();
        let mut input2_grad = DenseMatrix::default();
        let mut output = DenseMatrix::default();

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrices from CPU
        {
            input1.load_from_slice(shape, &[1.0; 6]);

            input2.load_from_slice(shape, &[2.0, 1.0, 4.0, 3.0, 0.0, 4.0]);

            assert_eq!(input1.shape(), shape);
            assert_eq!(input2.shape(), shape);

            util::panic_if_device_error("Failed to load data from CPU!");
        }

        // normal matmul
        {
            DenseMatrix::submatrix_product(&mut ctx, key_size, &input1, &input2, &mut output);

            util::panic_if_device_error("Failed to calculate matmul!");

            assert_eq!(output.shape(), Shape::new(4, 3));

            let mut buf = [0.0; 12];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, [2.0, 2.0, 1.0, 1.0, 4.0, 4.0, 3.0, 3.0, 0.0, 0.0, 4.0, 4.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // backprop normal matmul
        {
            DenseMatrix::backprop_submatrix_product(
                &mut ctx,
                key_size,
                &input1,
                Some(&mut input1_grad),
                &input2,
                Some(&mut input2_grad),
                &output,
            );

            util::panic_if_device_error("Failed to backprop matmul!");

            assert_eq!(input1_grad.shape(), shape);
            assert_eq!(input2_grad.shape(), shape);

            let mut grad1 = [0.0; 6];
            input1_grad.write_to_slice(&mut grad1);
            assert_eq!(grad1, [5.0, 5.0, 25.0, 25.0, 16.0, 16.0]);

            let mut grad2 = [0.0; 6];
            input2_grad.write_to_slice(&mut grad2);
            assert_eq!(grad2, [4.0, 2.0, 8.0, 6.0, 0.0, 8.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }
    }
}
