use crate::{backend::ops, Activation, DenseMatrix, SparseMatrix};
use bullet_core::shape::Shape;

pub fn affine_dual(
    input_a: &DenseMatrix,
    input_b1: &SparseMatrix,
    input_b2: &SparseMatrix,
    input_c: &DenseMatrix,
    output: &mut DenseMatrix,
    activation: Activation,
) {
    assert!(input_a.shape.batch_size().is_none());
    assert!(input_c.shape.batch_size().is_none());

    assert_eq!(input_b1.shape, input_b2.shape);
    assert_eq!(input_b1.shape.cols(), 1);
    assert_eq!(input_b1.nnz, input_b2.nnz);

    assert_eq!(input_c.shape.rows(), input_a.shape.rows());
    assert_eq!(input_c.shape.cols(), 1);

    let mut output_shape = input_a.shape * input_b1.shape;
    output_shape = Shape::from_raw(output_shape.rows() * 2, output_shape.cols(), output_shape.batch_size());

    output.reshape_if_needed(output_shape);

    unsafe {
        ops::sparseAffineDualForward(
            input_b1.shape.batch_size().unwrap_or(1),
            input_b1.nnz,
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
    input_b1: &SparseMatrix,
    input_b2: &SparseMatrix,
    input_c: &DenseMatrix,
    input_c_grad: &mut DenseMatrix,
    outputs: &DenseMatrix,
    output_grad: &DenseMatrix,
    activation: Activation,
) {
    assert!(input_a.shape.batch_size().is_none());
    assert!(input_c.shape.batch_size().is_none());

    assert_eq!(input_b1.shape, input_b2.shape);
    assert_eq!(input_b1.shape.cols(), 1);
    assert_eq!(input_b1.nnz, input_b2.nnz);
    assert_eq!(outputs.shape, output_grad.shape);

    input_a_grad.reshape_if_needed(input_a.shape());
    input_c_grad.reshape_if_needed(input_c.shape());

    unsafe {
        ops::sparseAffineDualBackward(
            input_b1.shape.batch_size().unwrap_or(1),
            input_b1.nnz,
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{backend::util, ExecutionContext};
    use bullet_core::shape::Shape;

    #[test]
    fn test_affine_dual() {
        let device = Arc::new(ExecutionContext::default());

        let shape1 = Shape::new(2, 3);
        let shape2 = Shape::new_batched(3, 1, 3);

        let mut input1 = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input2 = SparseMatrix::zeroed(device.clone(), Shape::new(1, 1), 1);
        let mut input3 = SparseMatrix::zeroed(device.clone(), Shape::new(1, 1), 1);
        let mut input4 = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input1_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input4_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

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

                input3.load_from_slice(shape2, 2, &[0, -1, 1, 1, -1, -1]);
            }

            input4.load_from_slice(Shape::new(2, 1), &[0.0, 0.0]);

            assert_eq!(input1.shape(), shape1);
            assert_eq!(input2.shape(), shape2);

            util::panic_if_device_error("Failed to load data from CPU!");
        }

        // sparse linear
        {
            affine_dual(&input1, &input2, &input3, &input4, &mut output, Activation::Identity);

            util::panic_if_device_error("Failed to calculate matmul!");

            assert_eq!(output.shape(), Shape::new_batched(4, 1, 3));

            let mut buf = [0.0; 12];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, [-1.0, 4.0, -1.0, 4.0, 2.0, -5.0, 4.0, -4.0, 0.0, 0.0, 0.0, 0.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // backprop sparse linear
        {
            backprop_affine_dual(
                &input1,
                &mut input1_grad,
                &input2,
                &input3,
                &input4,
                &mut input4_grad,
                &output,
                &output,
                Activation::Identity,
            );

            util::panic_if_device_error("Failed to backprop matmul!");

            assert_eq!(input1_grad.shape(), shape1);

            let mut grad1 = [0.0; 6];
            input1_grad.write_to_slice(&mut grad1);
            assert_eq!(grad1, [-2.0, 8.0, 10.0, -13.0, 2.0, -5.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }
    }
}
