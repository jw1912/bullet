use bullet_core::device::DeviceBuffer;

use crate::{backend::blas, DenseMatrix, Shape};

pub fn concat(input_a: &DenseMatrix, input_b: &DenseMatrix, output: &mut DenseMatrix) {
    let cols = input_a.shape.cols();
    assert_eq!(cols, input_b.shape.cols());

    let output_shape = Shape::new(input_a.shape.rows() + input_b.shape.rows(), cols);
    output.reshape_if_needed(output_shape);

    unsafe {
        blas::copy_strided(
            input_a.buf.device().as_ref(),
            input_a.shape.rows(),
            cols,
            input_a.shape.rows(),
            input_a.buf.ptr(),
            output_shape.rows(),
            output.buf.mut_ptr(),
            false,
        );

        blas::copy_strided(
            input_a.buf.device().as_ref(),
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
    input_a: &DenseMatrix,
    input_a_grad: Option<&mut DenseMatrix>,
    input_b: &DenseMatrix,
    input_b_grad: Option<&mut DenseMatrix>,
    output_grad: &DenseMatrix,
) {
    let cols = input_a.shape.cols();
    assert_eq!(cols, input_b.shape.cols());
    assert_eq!(cols, output_grad.shape.cols());

    if let Some(grad) = input_a_grad {
        grad.reshape_if_needed(input_a.shape);

        unsafe {
            blas::copy_strided(
                grad.buf.device().as_ref(),
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
            blas::copy_strided(
                grad.buf.device().as_ref(),
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{backend::util, ExecutionContext, Shape};

    #[test]
    fn test_concat() {
        let device = Arc::new(ExecutionContext::default());

        let shape1 = Shape::new(3, 3);
        let shape2 = Shape::new(1, 3);

        let mut input1 = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input2 = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input1_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input2_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

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
            concat(&input1, &input2, &mut output);

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

            backprop_concat(&input1, Some(&mut input1_grad), &input2, Some(&mut input2_grad), &output);

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
