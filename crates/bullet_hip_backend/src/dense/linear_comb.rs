use bullet_core::device::DeviceBuffer;

use crate::{backend::ops, Buffer, DeviceError};

pub fn linear_comb_single(
    size: usize,
    alpha: f32,
    input_a: Option<&Buffer<f32>>,
    beta: f32,
    input_b: Option<&Buffer<f32>>,
    output: &mut Buffer<f32>,
) -> Result<(), DeviceError> {
    match (input_a, input_b) {
        (None, None) => {
            if size > output.size() {
                return Err(DeviceError::ExpectedIllegalAddressAccess);
            }

            unsafe {
                ops::scale_assign(size, output.mut_ptr(), alpha);
            }
        }
        (None, Some(b)) => {
            if size > output.size() || size > b.size() {
                return Err(DeviceError::ExpectedIllegalAddressAccess);
            }

            unsafe {
                ops::scale_add_assign(size, alpha, output.mut_ptr(), beta, b.ptr());
            }
        }
        (Some(a), Some(b)) => {
            if size > output.size() || size > a.size() || size > b.size() {
                return Err(DeviceError::ExpectedIllegalAddressAccess);
            }

            unsafe {
                ops::linear_comb(size, alpha, a.ptr(), beta, b.ptr(), output.mut_ptr());
            }
        }
        (Some(a), None) => {
            if size > output.size() || size > a.size() {
                return Err(DeviceError::ExpectedIllegalAddressAccess);
            }

            unsafe {
                ops::scale(size, alpha, a.ptr(), output.mut_ptr());
            }
        }
    }

    Ok(())
}
