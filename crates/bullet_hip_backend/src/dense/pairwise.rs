use bullet_core::device::DeviceBuffer;

use crate::{
    backend::{ops, Buffer},
    DeviceError,
};

pub fn pairwise(
    single_size: usize,
    batch_size: usize,
    input: &Buffer<f32>,
    output: &mut Buffer<f32>,
) -> Result<(), DeviceError> {
    if single_size * batch_size > input.size() {
        return Err(DeviceError::ExpectedIllegalAddressAccess);
    }

    assert_eq!(single_size % 2, 0);

    if (single_size / 2) * batch_size > output.size() {
        return Err(DeviceError::ExpectedIllegalAddressAccess);
    }

    unsafe {
        ops::pairwiseMul(batch_size, single_size / 2, input.ptr(), output.mut_ptr());
    }

    Ok(())
}

pub fn backprop_pairwise(
    single_size: usize,
    batch_size: usize,
    input: &Buffer<f32>,
    output_grad: &Buffer<f32>,
    input_grad: &mut Buffer<f32>,
) -> Result<(), DeviceError> {
    if single_size * batch_size > input.size().max(input_grad.size()) {
        return Err(DeviceError::ExpectedIllegalAddressAccess);
    }

    assert_eq!(single_size % 2, 0);

    if (single_size / 2) * batch_size > output_grad.size() {
        return Err(DeviceError::ExpectedIllegalAddressAccess);
    }

    unsafe {
        ops::backpropPairwiseMul(batch_size, single_size / 2, input.ptr(), output_grad.ptr(), input_grad.mut_ptr());
    }

    Ok(())
}
