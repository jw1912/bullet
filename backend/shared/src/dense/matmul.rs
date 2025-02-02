use bullet_core::{device::DeviceBuffer, shape::Shape};

use crate::{backend::blas, DenseMatrix};

#[allow(clippy::too_many_arguments)]
fn sgemm(
    input_a: &DenseMatrix,
    shape_a: Shape,
    trans_a: bool,
    input_b: &DenseMatrix,
    shape_b: Shape,
    trans_b: bool,
    output: &mut DenseMatrix,
    output_shape: Shape,
    increment: bool,
) {
    assert!(shape_a.batch_size().is_none());
    assert!(shape_b.batch_size().is_none());

    let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);
    output.reshape_if_needed(output_shape);
    assert_eq!(output_shape.size(), shape_o.size());

    unsafe {
        blas::sgemm(
            input_a.buf.device().as_ref(),
            input_a.buf.ptr(),
            shape_a.rows(),
            shape_a.cols(),
            trans_a,
            input_b.buf.ptr(),
            shape_b.rows(),
            shape_b.cols(),
            trans_b,
            output.buf.mut_ptr(),
            shape_o.rows(),
            shape_o.cols(),
            increment,
        );
    }
}

fn sgemm_batched(
    input_a: &DenseMatrix,
    trans_a: bool,
    input_b: &DenseMatrix,
    trans_b: bool,
    output: &mut DenseMatrix,
    increment: bool,
) {
    assert_eq!(input_a.shape.batch_size(), input_b.shape.batch_size());

    let output_shape = input_a.shape.maybe_transpose(trans_a) * input_b.shape.maybe_transpose(trans_b);
    let batch_size = input_a.shape.batch_size().unwrap_or(1);

    unsafe {
        blas::batched_sgemm(
            input_a.buf.device().as_ref(),
            batch_size,
            input_a.buf.ptr(),
            input_a.shape.rows(),
            input_a.shape.cols(),
            trans_a,
            input_b.buf.ptr(),
            input_b.shape.rows(),
            input_b.shape.cols(),
            trans_b,
            output.buf.mut_ptr(),
            output_shape.rows(),
            output_shape.cols(),
            increment,
        );
    }
}

pub fn matmul(input_a: &DenseMatrix, trans_a: bool, input_b: &DenseMatrix, trans_b: bool, output: &mut DenseMatrix) {
    let output_shape = input_a.shape.maybe_transpose(trans_a) * input_b.shape.maybe_transpose(trans_b);

    match (input_a.shape.batch_size(), input_b.shape.batch_size()) {
        (Some(_), Some(_)) => sgemm_batched(input_a, trans_a, input_b, trans_b, output, false),
        (None, None) => {
            sgemm(input_a, input_a.shape, trans_a, input_b, input_b.shape, trans_b, output, output_shape, false);
        }
        (None, Some(x)) => {
            let shape_b = Shape::new(input_b.shape.rows(), x);
            if trans_b || input_b.shape.cols() > 1 {
                unimplemented!()
            }

            sgemm(input_a, input_a.shape, trans_a, input_b, shape_b, trans_b, output, output_shape, false);
        }
        (Some(_), None) => unimplemented!(),
    }
}

pub fn backprop_matmul(
    input_a: &DenseMatrix,
    input_a_grad: Option<&mut DenseMatrix>,
    trans_a: bool,
    input_b: &DenseMatrix,
    input_b_grad: Option<&mut DenseMatrix>,
    trans_b: bool,
    output_grad: &DenseMatrix,
) {
    match (input_a.shape.batch_size(), input_b.shape.batch_size()) {
        (Some(_), Some(_)) => {
            backprop_batched_matmul(input_a, input_a_grad, trans_a, input_b, input_b_grad, trans_b, output_grad);
        }
        (None, None) => {
            backprop_single_matmul(
                input_a,
                input_a.shape,
                input_a_grad,
                trans_a,
                input_b,
                input_b.shape,
                input_b_grad,
                trans_b,
                output_grad,
            );
        }
        (None, Some(x)) => {
            let shape_b = Shape::new(input_b.shape.rows(), x);
            if trans_b || input_b.shape.cols() > 1 {
                unimplemented!()
            }

            backprop_single_matmul(
                input_a,
                input_a.shape,
                input_a_grad,
                trans_a,
                input_b,
                shape_b,
                input_b_grad,
                trans_b,
                output_grad,
            );
        }
        (Some(_), None) => unimplemented!(),
    }
}

#[allow(clippy::too_many_arguments)]
fn backprop_single_matmul(
    input_a: &DenseMatrix,
    shape_a: Shape,
    input_a_grad: Option<&mut DenseMatrix>,
    trans_a: bool,
    input_b: &DenseMatrix,
    shape_b: Shape,
    input_b_grad: Option<&mut DenseMatrix>,
    trans_b: bool,
    output_grad: &DenseMatrix,
) {
    let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);

    if let Some(grad_a) = input_a_grad {
        if trans_a {
            sgemm(input_b, shape_b, trans_b, output_grad, shape_o, true, grad_a, input_a.shape, true);
        } else {
            sgemm(output_grad, shape_o, false, input_b, shape_b, !trans_b, grad_a, input_a.shape, true);
        }
    }

    if let Some(grad_b) = input_b_grad {
        if trans_b {
            sgemm(output_grad, shape_o, true, input_a, shape_a, trans_a, grad_b, input_b.shape, true);
        } else {
            sgemm(input_a, shape_a, !trans_a, output_grad, shape_o, false, grad_b, input_b.shape, true);
        }
    }
}

fn backprop_batched_matmul(
    input_a: &DenseMatrix,
    input_a_grad: Option<&mut DenseMatrix>,
    trans_a: bool,
    input_b: &DenseMatrix,
    input_b_grad: Option<&mut DenseMatrix>,
    trans_b: bool,
    output_grad: &DenseMatrix,
) {
    if let Some(grad_a) = input_a_grad {
        if trans_a {
            sgemm_batched(input_b, trans_b, output_grad, true, grad_a, true);
        } else {
            sgemm_batched(output_grad, false, input_b, !trans_b, grad_a, true);
        }
    }

    if let Some(grad_b) = input_b_grad {
        if trans_b {
            sgemm_batched(output_grad, true, input_a, trans_a, grad_b, true);
        } else {
            sgemm_batched(input_a, !trans_a, output_grad, false, grad_b, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{backend::util, ExecutionContext, Shape};

    #[test]
    fn single_matmul() {
        test_matmul_with_shape(Shape::new(3, 2));
    }

    #[test]
    fn batched_matmul() {
        test_matmul_with_shape(Shape::new_batched(3, 1, 2));
    }

    fn test_matmul_with_shape(shape2: Shape) {
        let device = Arc::new(ExecutionContext::default());

        let shape1 = Shape::new(2, 3);

        let mut input1 = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input2 = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input1_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input2_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
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

            input2.load_from_slice(
                shape2,
                // [ 1.0 ]
                // [ 2.0 ]
                // [ 3.0 ]
                &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            );

            assert_eq!(input1.shape(), shape1);
            assert_eq!(input2.shape(), shape2);

            util::panic_if_device_error("Failed to load data from CPU!");
        }

        // normal matmul
        {
            matmul(&input1, false, &input2, false, &mut output);

            util::panic_if_device_error("Failed to calculate matmul!");

            assert_eq!(output.shape(), Shape::from_raw(2, shape2.cols(), shape2.batch_size()));

            let mut buf = [0.0; 4];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, [3.0, -9.0, 3.0, -9.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // backprop normal matmul
        {
            backprop_matmul(&input1, Some(&mut input1_grad), false, &input2, Some(&mut input2_grad), false, &output);

            util::panic_if_device_error("Failed to backprop matmul!");

            assert_eq!(input1_grad.shape(), shape1);
            assert_eq!(input2_grad.shape(), shape2);

            let mut grad1 = [0.0; 6];
            input1_grad.write_to_slice(&mut grad1);
            assert_eq!(grad1, [6.0, -18.0, 12.0, -36.0, 18.0, -54.0]);

            let mut grad2 = [0.0; 6];
            input2_grad.write_to_slice(&mut grad2);
            assert_eq!(grad2, [-39.0, 24.0, 27.0, -39.0, 24.0, 27.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        if shape2.batch_size().is_none() {
            // transposed matmul
            {
                matmul(&input2, true, &input1, true, &mut output);

                util::panic_if_device_error("Failed to calculate transposed matmul!");

                assert_eq!(output.shape(), Shape::new(2, 2));

                let mut buf = [0.0; 4];
                output.write_to_slice(&mut buf);
                assert_eq!(buf, [3.0, 3.0, -9.0, -9.0]);

                util::panic_if_device_error("Failed to write data to CPU!");
            }

            // backprop transposed matmul
            {
                input1_grad.set_zero();
                input2_grad.set_zero();

                backprop_matmul(&input2, Some(&mut input2_grad), true, &input1, Some(&mut input1_grad), true, &output);

                util::panic_if_device_error("Failed to backprop transposed matmul!");

                assert_eq!(input1_grad.shape(), shape1);
                assert_eq!(input2_grad.shape(), shape2);

                let mut grad1 = [0.0; 6];
                input1_grad.write_to_slice(&mut grad1);
                assert_eq!(grad1, [6.0, -18.0, 12.0, -36.0, 18.0, -54.0]);

                let mut grad2 = [0.0; 6];
                input2_grad.write_to_slice(&mut grad2);
                assert_eq!(grad2, [-39.0, 24.0, 27.0, -39.0, 24.0, 27.0]);

                util::panic_if_device_error("Failed to write data to CPU!");
            }
        }
    }
}
