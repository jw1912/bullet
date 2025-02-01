use bullet_core::device::DeviceBuffer;

use crate::{backend::blas, DenseMatrix, Shape};

#[allow(clippy::too_many_arguments)]
fn batched_sgemm(
    input_a: &DenseMatrix,
    shape_a: Shape,
    trans_a: bool,
    input_b: &DenseMatrix,
    shape_b: Shape,
    trans_b: bool,
    output: &mut DenseMatrix,
    increment: bool,
) {
    assert_eq!(shape_a.size(), input_a.shape.rows());
    assert_eq!(shape_b.size(), input_b.shape.rows());
    assert_eq!(input_a.shape.cols(), input_b.shape.cols());

    let output_shape = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);
    let batch_size = input_a.shape.cols();

    output.reshape_if_needed(Shape::new(output_shape.size(), batch_size));

    unsafe {
        blas::batched_sgemm(
            input_a.buf.device().as_ref(),
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

pub fn submatrix_product(key_size: usize, input_a: &DenseMatrix, input_b: &DenseMatrix, output: &mut DenseMatrix) {
    assert_eq!(input_a.shape.cols(), input_b.shape.cols());
    assert_eq!(input_a.shape.rows() % key_size, 0);

    let shape_a = Shape::new(key_size, input_a.shape.rows() / key_size);
    let shape_b = Shape::new(key_size, input_b.shape.rows() / key_size);

    batched_sgemm(input_a, shape_a, true, input_b, shape_b, false, output, false);
}

pub fn backprop_submatrix_product(
    key_size: usize,
    input_a: &DenseMatrix,
    input_a_grad: Option<&mut DenseMatrix>,
    input_b: &DenseMatrix,
    input_b_grad: Option<&mut DenseMatrix>,
    output_grad: &DenseMatrix,
) {
    assert_eq!(input_a.shape.cols(), input_b.shape.cols());
    assert_eq!(input_a.shape.rows() % key_size, 0);

    let shape_a = Shape::new(key_size, input_a.shape.rows() / key_size);
    let shape_b = Shape::new(key_size, input_b.shape.rows() / key_size);
    let output_shape = shape_a.transpose() * shape_b;

    assert_eq!(output_grad.shape.rows(), output_shape.size());

    if let Some(grad_a) = input_a_grad {
        batched_sgemm(input_b, shape_b, false, output_grad, output_shape, true, grad_a, true);
    }

    if let Some(grad_b) = input_b_grad {
        batched_sgemm(input_a, shape_a, false, output_grad, output_shape, false, grad_b, true);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{backend::util, ExecutionContext, Shape};

    #[test]
    fn test_submatrix_product() {
        let device = Arc::new(ExecutionContext::default());

        let shape = Shape::new(2, 3);
        let key_size = 1;

        let mut input1 = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input2 = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input1_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input2_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

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
            submatrix_product(key_size, &input1, &input2, &mut output);

            util::panic_if_device_error("Failed to calculate matmul!");

            assert_eq!(output.shape(), Shape::new(4, 3));

            let mut buf = [0.0; 12];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, [2.0, 2.0, 1.0, 1.0, 4.0, 4.0, 3.0, 3.0, 0.0, 0.0, 4.0, 4.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // backprop normal matmul
        {
            backprop_submatrix_product(
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
