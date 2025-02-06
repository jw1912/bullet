use bullet_core::device::DeviceBuffer;

use crate::backend::{ops, Buffer};

#[allow(clippy::too_many_arguments)]
pub fn adamw(
    size: usize,
    params: &mut Buffer<f32>,
    gradient: &Buffer<f32>,
    momentum: &mut Buffer<f32>,
    velocity: &mut Buffer<f32>,
    beta1: f32,
    beta2: f32,
    min_weight: f32,
    max_weight: f32,
    decay: f32,
    gradient_factor: f32,
    learning_rate: f32,
) {
    assert!(size <= params.size());
    assert!(size <= gradient.size());
    assert!(size <= momentum.size());
    assert!(size <= velocity.size());

    let decay = 1.0 - learning_rate * decay;

    unsafe {
        ops::AdamW(
            size,
            decay,
            beta1,
            beta2,
            min_weight,
            max_weight,
            gradient_factor,
            learning_rate,
            params.mut_ptr(),
            momentum.mut_ptr(),
            velocity.mut_ptr(),
            gradient.ptr(),
        );
    }
}
