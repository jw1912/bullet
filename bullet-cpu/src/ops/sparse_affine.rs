use bullet_core::Feat;

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
