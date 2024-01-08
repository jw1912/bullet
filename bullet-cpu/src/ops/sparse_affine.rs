use bullet_core::Feat;

use crate::DeviceHandles;

pub unsafe fn sparse_affine_forward(
    handle: DeviceHandles,
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

pub unsafe fn sparse_affine_backward(
    handle: DeviceHandles,
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
