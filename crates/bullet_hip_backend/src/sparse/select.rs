use bullet_core::device::DeviceBuffer;

use crate::{
    backend::{ops, Buffer},
    OperationResult,
};

pub fn select(
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    input: &Buffer<f32>,
    indices: &Buffer<i32>,
    output: &mut Buffer<f32>,
) -> OperationResult {
    assert!(batch_size * input_size <= input.size());
    assert!(batch_size <= indices.size());
    assert!(batch_size * output_size <= output.size());

    unsafe {
        ops::selectForward(batch_size, input_size, output_size, indices.ptr(), input.ptr(), output.mut_ptr());
    }

    Ok(())
}

pub fn select_backprop(
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    indices: &Buffer<i32>,
    output_grad: &Buffer<f32>,
    input_grad: &mut Buffer<f32>,
) -> OperationResult {
    assert!(batch_size * input_size <= input_grad.size());
    assert!(batch_size <= indices.size());
    assert!(batch_size * output_size <= output_grad.size());

    unsafe {
        ops::selectBackprop(
            batch_size,
            input_size,
            output_size,
            indices.ptr(),
            output_grad.ptr(),
            input_grad.mut_ptr(),
        );
    }

    Ok(())
}
