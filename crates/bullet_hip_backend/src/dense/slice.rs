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
