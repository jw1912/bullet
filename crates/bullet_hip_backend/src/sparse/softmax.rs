use crate::backend::{ops, Buffer};

pub fn softmax_across_batch_masked(
    batch_size: usize,
    single_size: usize,
    nnz: usize,
    masks: &Buffer<i32>,
    input: &Buffer<f32>,
    output: &mut Buffer<f32>,
) {
    unsafe {
        ops::softmax_across_columns_masked(nnz, single_size, batch_size, masks.ptr(), input.ptr(), output.mut_ptr());
    }
}

#[allow(clippy::too_many_arguments)]
pub fn crossentropy_masked(
    batch_size: usize,
    _: usize,
    nnz: usize,
    masks: &Buffer<i32>,
    pred: &Buffer<f32>,
    target: &Buffer<f32>,
    output: &mut Buffer<f32>,
    error: &mut Buffer<f32>,
) {
    unsafe {
        ops::crossentropy_masked(
            nnz,
            batch_size,
            masks.ptr(),
            pred.ptr(),
            target.ptr(),
            output.mut_ptr(),
            error.mut_ptr(),
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn backprop_softmax_crossentropy_masked(
    batch_size: usize,
    single_size: usize,
    nnz: usize,
    masks: &Buffer<i32>,
    softmaxed: &Buffer<f32>,
    target: &Buffer<f32>,
    output_grad: &Buffer<f32>,
    input_grad: &mut Buffer<f32>,
) {
    unsafe {
        ops::backprop_softmax_crossentropy_masked(
            nnz,
            single_size,
            batch_size,
            masks.ptr(),
            softmaxed.ptr(),
            target.ptr(),
            output_grad.ptr(),
            input_grad.mut_ptr(),
        );
    }
}
