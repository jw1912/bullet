use crate::{backend::ops, DenseMatrix};

#[allow(clippy::too_many_arguments)]
pub fn adamw(
    params: &mut DenseMatrix,
    gradient: &DenseMatrix,
    momentum: &mut DenseMatrix,
    velocity: &mut DenseMatrix,
    beta1: f32,
    beta2: f32,
    min_weight: f32,
    max_weight: f32,
    decay: f32,
    gradient_factor: f32,
    learning_rate: f32,
) {
    assert!(params.shape.batch_size().is_none());
    assert_eq!(params.shape, gradient.shape);
    assert_eq!(params.shape, momentum.shape);
    assert_eq!(params.shape, velocity.shape);

    let decay = 1.0 - learning_rate * decay;

    unsafe {
        ops::AdamW(
            params.shape.size(),
            decay,
            beta1,
            beta2,
            min_weight,
            max_weight,
            gradient_factor,
            learning_rate,
            params.buf.mut_ptr(),
            momentum.buf.mut_ptr(),
            velocity.buf.mut_ptr(),
            gradient.buf.ptr(),
        );
    }
}
