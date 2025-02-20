use bullet_core::device::{DeviceBuffer, OperationError};

use crate::{
    backend::{ops, Buffer},
    OperationResult,
};

pub fn pairwise(
    mut single_size: usize,
    mut batch_size: usize,
    input: &Buffer<f32>,
    output: &mut Buffer<f32>,
    post_concat: bool,
) -> OperationResult {
    if single_size * batch_size > input.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    if post_concat {
        assert_eq!(single_size % 2, 0);
        single_size /= 2;
        batch_size *= 2;
    }

    assert_eq!(single_size % 2, 0);

    if (single_size / 2) * batch_size > output.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    unsafe {
        ops::pairwiseMul(batch_size, single_size / 2, input.ptr(), output.mut_ptr());
    }

    Ok(())
}

pub fn backprop_pairwise(
    mut single_size: usize,
    mut batch_size: usize,
    input: &Buffer<f32>,
    output_grad: &Buffer<f32>,
    input_grad: &mut Buffer<f32>,
    post_concat: bool,
) -> OperationResult {
    if single_size * batch_size > input.size().max(input_grad.size()) {
        return Err(OperationError::IndexOutOfBounds);
    }

    if post_concat {
        assert_eq!(single_size % 2, 0);
        single_size /= 2;
        batch_size *= 2;
    }

    assert_eq!(single_size % 2, 0);

    if (single_size / 2) * batch_size > output_grad.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    unsafe {
        ops::backpropPairwiseMul(batch_size, single_size / 2, input.ptr(), output_grad.ptr(), input_grad.mut_ptr());
    }

    Ok(())
}
