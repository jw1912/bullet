use bullet_core::device::{DeviceBuffer, OperationError};

use crate::{backend::ops, Buffer, OperationResult};

pub fn softmax_across_batch(
    batch_size: usize,
    single_size: usize,
    input: &Buffer<f32>,
    output: &mut Buffer<f32>,
) -> OperationResult {
    assert!(batch_size * single_size <= input.size());
    assert!(batch_size * single_size <= output.size());

    unsafe {
        ops::softmax_across_columns(single_size, batch_size, input.ptr(), output.mut_ptr());
    }

    Ok(())
}

pub fn crossentropy(
    size: usize,
    pred: &Buffer<f32>,
    target: &Buffer<f32>,
    output: &mut Buffer<f32>,
) -> OperationResult {
    if size > pred.size() || size > target.size() || size > output.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    unsafe {
        ops::crossentropy(size, pred.ptr(), target.ptr(), output.mut_ptr());
    }

    Ok(())
}

pub fn backprop_softmax_crossentropy(
    size: usize,
    softmaxed: &Buffer<f32>,
    target: &Buffer<f32>,
    output_grad: &Buffer<f32>,
    input_grad: &mut Buffer<f32>,
) -> OperationResult {
    if size > softmaxed.size() || size > target.size() || size > output_grad.size() || size > input_grad.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    unsafe {
        ops::backprop_softmax_cross_entropy(
            size,
            softmaxed.ptr(),
            target.ptr(),
            output_grad.ptr(),
            input_grad.mut_ptr(),
        );
    }

    Ok(())
}
