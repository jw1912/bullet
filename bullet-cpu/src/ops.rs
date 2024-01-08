#![allow(unused_variables)]
use bullet_core::Feat;

use crate::DeviceHandles;

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn activate_relu(size: usize, inp: *const f32, out: *mut f32) {
    unimplemented!();
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn activate_crelu(size: usize, inp: *const f32, out: *mut f32) {
    unimplemented!();
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn activate_screlu(size: usize, inp: *const f32, out: *mut f32) {
    unimplemented!();
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn backprop_relu(size: usize, inp: *const f32, out: *mut f32) {
    unimplemented!();
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn backprop_crelu(size: usize, inp: *const f32, out: *mut f32) {
    unimplemented!();
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn backprop_screlu(size: usize, inp: *const f32, out: *mut f32) {
    unimplemented!();
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn sigmoid_mse(
    buffer_size: usize,
    outputs: *mut f32,
    results: *const f32,
    error: *mut f32,
) {
    unimplemented!();
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn sparse_affine_backward(
    batch_size: usize,
    max_input_size: usize,
    output_size: usize,
    weights_grad: *mut f32,
    biases_grad: *mut f32,
    inputs: *const Feat,
    errors: *const f32,
) {
    unimplemented!();
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn sparse_affine_forward(
    batch_size: usize,
    max_input_size: usize,
    output_size: usize,
    weights: *const f32,
    biases: *const f32,
    inputs: *const Feat,
    outputs: *mut f32,
) {
    unimplemented!();
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn splat_add(
    batch_size: usize,
    tensor_size: usize,
    inp: *const f32,
    out: *mut f32,
) {
    unimplemented!();
}

/// # Safety
/// Pass valid pointers and sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn update_weights(
    network_size: usize,
    decay: f32,
    adj: f32,
    rate: f32,
    network: *mut f32,
    momentum: *mut f32,
    velocity: *mut f32,
    gradients: *const f32,
) {
    unimplemented!();
}

/// # Safety
/// This should only be used and exposed internally.
pub unsafe fn splat_mul_matrix_vector(
    handle: DeviceHandles,
    m: usize,
    n: usize,
    a_ptr: *const f32,
    x_ptr: *const f32,
    y_ptr: *mut f32,
    batch_size: usize,
) {
    unimplemented!();
}

/// # Safety
/// This should only be used and exposed internally.
pub unsafe fn splat_mul_matrixt_vector(
    handle: DeviceHandles,
    m: usize,
    n: usize,
    a_ptr: *const f32,
    y_ptr: *const f32,
    x_ptr: *mut f32,
    batch_size: usize,
) {
    unimplemented!();
}

/// # Safety
/// This should only be used and exposed internally.
pub unsafe fn reduce_add_mul_vector_vectort(
    handle: DeviceHandles,
    m: usize,
    n: usize,
    y_ptr: *const f32,
    x_ptr: *const f32,
    a_ptr: *mut f32,
    batch_size: usize,
) {
    unimplemented!();
}

/// # Safety
/// This should only be used and exposed internally.
pub unsafe fn reduce_add(
    handle: DeviceHandles,
    ones: *const f32,
    batch_size: usize,
    out_size: usize,
    inp: *const f32,
    out: *mut f32,
) {
    unimplemented!();
}
