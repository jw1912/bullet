use bullet_core::device::{DeviceBuffer, OperationError};

use crate::{
    backend::{ops, Buffer},
    OperationResult,
};

pub fn abs_power_error(
    power: f32,
    size: usize,
    input_a: &Buffer<f32>,
    input_b: &Buffer<f32>,
    output: &mut Buffer<f32>,
) -> OperationResult {
    if size > input_a.size() || size > input_b.size() || size > output.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    unsafe {
        ops::powerError(size, input_a.ptr(), input_b.ptr(), output.mut_ptr(), power);
    }

    Ok(())
}

pub fn backprop_abs_power_error_single(
    power: f32,
    size: usize,
    input_a: &Buffer<f32>,
    input_b: &Buffer<f32>,
    output_grad: &Buffer<f32>,
    input_a_grad: &mut Buffer<f32>,
) -> OperationResult {
    if size > input_a.size() || size > input_b.size() || size > output_grad.size() || size > input_a_grad.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    unsafe {
        ops::backpropPowerError(size, input_a.ptr(), input_b.ptr(), output_grad.ptr(), input_a_grad.mut_ptr(), power);
    }

    Ok(())
}
