use bullet_core::device::DeviceBuffer;

use crate::{backend::blas, DenseMatrix, Shape};

pub fn slice_vector_batched(input: &DenseMatrix, start: usize, end: usize, output: &mut DenseMatrix) {
    assert_eq!(input.shape.cols(), 1);
    assert!(end > start, "Invalid slice indices! end = {end} > start = {start}");
    assert!(
        end <= input.shape.rows(),
        "Slice index out of bounds! Number of rows is {} but slice endpoint is {end}!",
        input.shape.rows()
    );

    let output_shape = Shape::from_raw(end - start, 1, input.shape.batch_size());
    output.reshape_if_needed(output_shape);

    unsafe {
        blas::copy_strided(
            input.buf.device().as_ref(),
            output_shape.rows(),
            input.shape.batch_size().unwrap_or(1),
            input.shape.rows(),
            input.buf.ptr().add(start),
            output_shape.rows(),
            output.buf.mut_ptr(),
            false,
        );
    }
}

pub fn backprop_slice_vector_batched(
    input: &DenseMatrix,
    input_grad: Option<&mut DenseMatrix>,
    start: usize,
    end: usize,
    output_grad: &DenseMatrix,
) {
    assert_eq!(input.shape.cols(), 1);
    assert!(end > start, "Invalid slice indices! end = {end} > start = {start}");
    assert!(
        end <= input.shape.rows(),
        "Slice index out of bounds! Number of rows is {} but slice endpoint is {end}!",
        input.shape.rows()
    );
    let output_shape = Shape::from_raw(end - start, 1, input.shape.batch_size());
    assert_eq!(output_shape, output_grad.shape);

    if let Some(grad) = input_grad {
        grad.reshape_if_needed(input.shape);

        unsafe {
            blas::copy_strided(
                output_grad.buf.device().as_ref(),
                output_shape.rows(),
                output_shape.batch_size().unwrap_or(1),
                output_shape.rows(),
                output_grad.buf.ptr(),
                grad.shape.rows(),
                grad.buf.mut_ptr().add(start),
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
    fn slice() {
        let device = Arc::new(ExecutionContext::default());

        let shape = Shape::new_batched(3, 1, 3);

        let mut input = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrices from CPU
        {
            input.load_from_slice(shape, &[-1.0, 4.0, 2.0, -2.0, 0.0, -3.0, 1.0, 1.0, 1.0]);

            assert_eq!(input.shape(), shape);

            util::panic_if_device_error("Failed to load data from CPU!");
        }

        // concat
        {
            slice_vector_batched(&input, 0, 2, &mut output);

            util::panic_if_device_error("Failed to concat matrices!");

            assert_eq!(output.shape(), Shape::new_batched(2, 1, 3));

            let mut buf = [0.0; 6];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, [-1.0, 4.0, -2.0, 0.0, 1.0, 1.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // de-concat
        {
            input_grad.load_from_slice(shape, &[1.0; 9]);

            backprop_slice_vector_batched(&input, Some(&mut input_grad), 0, 2, &output);

            util::panic_if_device_error("Failed to backprop slice!");

            assert_eq!(input_grad.shape(), shape);

            let mut grad = [0.0; 9];
            input_grad.write_to_slice(&mut grad);
            assert_eq!(grad, [0.0, 5.0, 1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 1.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }
    }
}
