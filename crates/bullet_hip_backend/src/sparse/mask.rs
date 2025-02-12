use bullet_core::device::DeviceBuffer;

use crate::backend::{ops, Buffer};

pub fn mask(
    batch_size: usize,
    single_size: usize,
    inputs: &Buffer<f32>,
    masks: &Buffer<i32>,
    nnz: usize,
    outputs: &mut Buffer<f32>,
) {
    assert!(batch_size * single_size <= inputs.size());
    assert!(batch_size * single_size <= outputs.size());
    assert!(batch_size * nnz <= masks.size());

    outputs.set_zero();

    unsafe {
        ops::sparse_mask(single_size, batch_size, nnz, inputs.ptr(), masks.ptr(), outputs.mut_ptr());
    }
}

pub fn backprop_mask(
    batch_size: usize,
    single_size: usize,
    output_grads: &Buffer<f32>,
    masks: &Buffer<i32>,
    nnz: usize,
    input_grads: &mut Buffer<f32>,
) {
    assert!(batch_size * single_size <= output_grads.size());
    assert!(batch_size * single_size <= input_grads.size());
    assert!(batch_size * nnz <= masks.size());

    unsafe {
        ops::sparse_mask_backprop(single_size, batch_size, nnz, output_grads.ptr(), masks.ptr(), input_grads.mut_ptr());
    }
}
