use bullet_core::device::{base::AdamConfig, DeviceBuffer};

use crate::{
    backend::{ops, Buffer},
    DeviceError,
};

#[allow(clippy::too_many_arguments)]
pub fn adam(
    size: usize,
    params: &mut Buffer<f32>,
    gradient: &Buffer<f32>,
    momentum: &mut Buffer<f32>,
    velocity: &mut Buffer<f32>,
    config: &AdamConfig,
) -> Result<(), DeviceError> {
    if size > params.size() || size > gradient.size() || size > momentum.size() || size > velocity.size() {
        return Err(DeviceError::ExpectedIllegalAddressAccess);
    }

    let (min, max) = config.clip.unwrap_or((f32::MIN, f32::MAX));

    unsafe {
        ops::Adam(
            size,
            config.beta1,
            config.beta2,
            config.gradient_factor,
            config.learning_rate,
            config.denom,
            config.decay,
            min,
            max,
            params.mut_ptr(),
            momentum.mut_ptr(),
            velocity.mut_ptr(),
            gradient.ptr(),
        );
    }

    Ok(())
}

pub fn clip(size: usize, params: &mut Buffer<f32>, min: f32, max: f32) -> Result<(), DeviceError> {
    if size > params.size() {
        return Err(DeviceError::ExpectedIllegalAddressAccess);
    }

    unsafe {
        ops::clip(size, params.mut_ptr(), min, max);
    }

    Ok(())
}
