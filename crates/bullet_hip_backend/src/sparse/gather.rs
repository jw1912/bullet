use bullet_core::device::DeviceBuffer;

use crate::backend::{ops, Buffer};

pub fn gather(
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    inputs: &Buffer<f32>,
    indices: &Buffer<i32>,
    outputs: &mut Buffer<f32>,
) {
    outputs.set_zero();

    unsafe {
        ops::gather(input_size, output_size, batch_size, inputs.ptr(), indices.ptr(), outputs.mut_ptr());
    }
}

pub fn backprop_gather(
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    output_grads: &Buffer<f32>,
    indices: &Buffer<i32>,
    input_grads: &mut Buffer<f32>,
) {
    unsafe {
        ops::gather_backprop(
            input_size,
            output_size,
            batch_size,
            output_grads.ptr(),
            indices.ptr(),
            input_grads.mut_ptr(),
        );
    }
}
