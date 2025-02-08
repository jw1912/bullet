use bullet_core::device::DeviceBuffer;

use crate::backend::{ops, Buffer};

pub fn abs_power_error(
    power: f32,
    size: usize,
    input_a: &Buffer<f32>,
    input_b: &Buffer<f32>,
    output: &mut Buffer<f32>,
) {
    assert!(size <= input_a.size());
    assert!(size <= input_b.size());
    assert!(size <= output.size());

    unsafe {
        ops::powerError(size, input_a.ptr(), input_b.ptr(), output.mut_ptr(), power);
    }
}

pub fn backprop_abs_power_error_single(
    power: f32,
    size: usize,
    input_a: &Buffer<f32>,
    input_b: &Buffer<f32>,
    output_grad: &Buffer<f32>,
    input_a_grad: &mut Buffer<f32>,
) {
    assert!(size <= input_a.size());
    assert!(size <= input_b.size());
    assert!(size <= output_grad.size());
    assert!(size <= input_a_grad.size());

    unsafe {
        ops::backpropPowerError(size, input_a.ptr(), input_b.ptr(), output_grad.ptr(), input_a_grad.mut_ptr(), power);
    }
}
