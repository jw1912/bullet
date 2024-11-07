use crate::{backend::ops, tensor::DenseMatrix, Activation, Shape};

use super::SparseMatrix;

impl SparseMatrix {
    pub fn affine_dual(
        input_a: &DenseMatrix,
        input_b1: &Self,
        input_b2: &Self,
        input_c: &DenseMatrix,
        output: &mut DenseMatrix,
        activation: Activation,
    ) {
        assert_eq!(input_b1.shape, input_b2.shape);
        assert_eq!(input_b1.max_active, input_b2.max_active);
        assert_eq!(input_c.shape.rows(), input_a.shape.rows());
        assert_eq!(input_c.shape.cols(), 1);

        let mut output_shape = input_a.shape * input_b1.shape;
        output_shape = Shape::new(output_shape.rows() * 2, output_shape.cols());

        output.reshape_if_needed(output_shape);

        unsafe {
            ops::sparseAffineDualForward(
                input_b1.shape.cols(),
                input_b1.max_active,
                input_a.shape().rows(),
                input_a.buf.ptr(),
                input_c.buf.ptr(),
                input_b1.buf.ptr(),
                input_b2.buf.ptr(),
                output.buf.mut_ptr(),
                activation as i32,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backprop_affine_dual(
        input_a: &DenseMatrix,
        input_a_grad: &mut DenseMatrix,
        input_b1: &Self,
        input_b2: &Self,
        input_c_grad: &mut DenseMatrix,
        outputs: &DenseMatrix,
        output_grad: &DenseMatrix,
        activation: Activation,
    ) {
        assert_eq!(input_b1.shape, input_b2.shape);
        assert_eq!(input_b1.max_active, input_b2.max_active);
        assert_eq!(outputs.shape, output_grad.shape);

        input_a_grad.reshape_if_needed(input_a.shape());

        unsafe {
            ops::sparseAffineDualBackward(
                input_b1.shape.cols(),
                input_b1.max_active,
                input_a.shape.rows(),
                input_a_grad.buf.mut_ptr(),
                input_c_grad.buf.mut_ptr(),
                input_b1.buf.ptr(),
                input_b2.buf.ptr(),
                outputs.buf.ptr(),
                output_grad.buf.ptr(),
                activation as i32,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::util, tensor::Shape};

    #[test]
    fn affine_dual() {
        let shape1 = Shape::new(2, 3);
        let shape2 = Shape::new(3, 3);

        let mut input1 = DenseMatrix::default();
        let mut input2 = SparseMatrix::default();
        let mut input3 = SparseMatrix::default();
        let mut input4 = DenseMatrix::default();
        let mut input1_grad = DenseMatrix::default();
        let mut input4_grad = DenseMatrix::default();
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

            unsafe {
                input2.load_from_slice(shape2, 2, &[0, -1, 1, 2, -1, -1]);

                input3.load_from_slice(shape2, 2, &[0, -1, 1, 2, -1, -1]);
            }

            input4.load_from_slice(Shape::new(2, 1), &[0.0, 0.0]);

            assert_eq!(input1.shape(), shape1);
            assert_eq!(input2.shape(), shape2);

            util::panic_if_device_error("Failed to load data from CPU!");
        }

        // sparse linear
        {
            SparseMatrix::affine_dual(&input1, &input2, &input3, &input4, &mut output, Activation::Identity);

            util::panic_if_device_error("Failed to calculate matmul!");

            assert_eq!(output.shape(), Shape::new(4, 3));

            let mut buf = [0.0; 12];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, [-1.0, 4.0, -1.0, 4.0, 2.0, -5.0, 2.0, -5.0, 0.0, 0.0, 0.0, 0.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // backprop sparse linear
        {
            SparseMatrix::backprop_affine_dual(
                &input1,
                &mut input1_grad,
                &input2,
                &input3,
                &mut input4_grad,
                &output,
                &output,
                Activation::Identity,
            );

            util::panic_if_device_error("Failed to backprop matmul!");

            assert_eq!(input1_grad.shape(), shape1);

            let mut grad1 = [0.0; 6];
            input1_grad.write_to_slice(&mut grad1);
            assert_eq!(grad1, [-2.0, 8.0, 4.0, -10.0, 4.0, -10.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }
    }
}
