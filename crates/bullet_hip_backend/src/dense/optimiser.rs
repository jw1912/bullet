use bullet_core::device::DeviceBuffer;

use crate::backend::{ops, Buffer};

#[allow(clippy::too_many_arguments)]
pub fn adam(
    size: usize,
    params: &mut Buffer<f32>,
    gradient: &Buffer<f32>,
    momentum: &mut Buffer<f32>,
    velocity: &mut Buffer<f32>,
    beta1: f32,
    beta2: f32,
    gradient_factor: f32,
    learning_rate: f32,
    denom: bool,
) {
    assert!(size <= params.size());
    assert!(size <= gradient.size());
    assert!(size <= momentum.size());
    assert!(size <= velocity.size());

    unsafe {
        ops::Adam(
            size,
            beta1,
            beta2,
            gradient_factor,
            learning_rate,
            denom,
            params.mut_ptr(),
            momentum.mut_ptr(),
            velocity.mut_ptr(),
            gradient.ptr(),
        );
    }
}

pub fn clip(size: usize, params: &mut Buffer<f32>, min: f32, max: f32) {
    assert!(size <= params.size());

    unsafe {
        ops::Clip(size, params.mut_ptr(), min, max);
    }
}
