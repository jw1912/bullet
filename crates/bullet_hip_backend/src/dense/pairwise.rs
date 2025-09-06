use acyclib::device::DeviceBuffer;

use crate::{
    DeviceError,
    backend::{Buffer, ops},
};

pub fn pairwise(
    offset: usize,
    stride: usize,
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
        ops::pairwiseMul(stride, batch_size, single_size / 2, input.ptr(), output.mut_ptr().add(offset));
    }

    Ok(())
}

pub fn backprop_pairwise(
    offset: usize,
    stride: usize,
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
        ops::backpropPairwiseMul(
            stride,
            batch_size,
            single_size / 2,
            input.ptr(),
            output_grad.ptr().add(offset),
            input_grad.mut_ptr(),
        );
    }

    Ok(())
}
