use bullet_core::device::DeviceBuffer;

use crate::{backend::ops, Buffer};

pub fn softmax_across_batch(batch_size: usize, single_size: usize, input: &Buffer<f32>, output: &mut Buffer<f32>) {
    assert!(batch_size * single_size <= input.size());
    assert!(batch_size * single_size <= output.size());

    unsafe {
        ops::softmax_across_columns(single_size, batch_size, input.ptr(), output.mut_ptr());
    }
}

pub fn crossentropy(size: usize, pred: &Buffer<f32>, target: &Buffer<f32>, output: &mut Buffer<f32>) {
    assert!(size <= pred.size());
    assert!(size <= target.size());
    assert!(size <= output.size());

    unsafe {
        ops::crossentropy(size, pred.ptr(), target.ptr(), output.mut_ptr());
    }
}

pub fn backprop_softmax_crossentropy(
    size: usize,
    softmaxed: &Buffer<f32>,
    target: &Buffer<f32>,
    output_grad: &Buffer<f32>,
    input_grad: &mut Buffer<f32>,
) {
    assert!(size <= softmaxed.size());
    assert!(size <= target.size());
    assert!(size <= output_grad.size());
    assert!(size <= input_grad.size());

    unsafe {
        ops::backprop_softmax_cross_entropy(
            size,
            softmaxed.ptr(),
            target.ptr(),
            output_grad.ptr(),
            input_grad.mut_ptr(),
        );
    }
}
