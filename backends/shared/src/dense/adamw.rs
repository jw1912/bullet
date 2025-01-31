use crate::{optimiser::AdamWParams, tensor::backend::ops};

use super::DenseMatrix;

impl DenseMatrix {
    pub fn adamw(
        &mut self,
        gradient: &Self,
        momentum: &mut Self,
        velocity: &mut Self,
        params: &AdamWParams,
        gradient_factor: f32,
        learning_rate: f32,
    ) {
        assert_eq!(self.shape, gradient.shape);
        assert_eq!(self.shape, momentum.shape);
        assert_eq!(self.shape, velocity.shape);

        let decay = 1.0 - learning_rate * params.decay;

        unsafe {
            ops::AdamW(
                self.shape.size(),
                decay,
                params.beta1,
                params.beta2,
                params.min_weight,
                params.max_weight,
                gradient_factor,
                learning_rate,
                self.buf.mut_ptr(),
                momentum.buf.mut_ptr(),
                velocity.buf.mut_ptr(),
                gradient.buf.ptr(),
            );
        }
    }
}
