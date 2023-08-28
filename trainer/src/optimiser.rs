use crate::arch::{NNUEParams, HIDDEN, INPUT};

pub trait Optimiser: Default {
    fn update_feature_weights(
        &mut self,
        params: &mut [f32; INPUT * HIDDEN],
        grads: &[f32; INPUT * HIDDEN],
        adj: f32,
        rate: f32,
    );
    fn update_feature_bias(
        &mut self,
        params: &mut [f32; HIDDEN],
        grads: &[f32; HIDDEN],
        adj: f32,
        rate: f32,
    );
    fn update_output_weights(
        &mut self,
        params: &mut [f32; 2 * HIDDEN],
        grads: &[f32; 2 * HIDDEN],
        adj: f32,
        rate: f32,
    );
    fn update_output_bias(&mut self, param: &mut f32, grad: f32, adj: f32, rate: f32);

    fn update_weights(&mut self, nnue: &mut NNUEParams, grads: &NNUEParams, adj: f32, rate: f32) {
        self.update_feature_weights(&mut nnue.feature_weights, &grads.feature_weights, adj, rate);
        self.update_output_weights(&mut nnue.output_weights, &grads.output_weights, adj, rate);
        self.update_feature_bias(&mut nnue.feature_bias, &grads.feature_bias, adj, rate);
        self.update_output_bias(&mut nnue.output_bias, grads.output_bias, adj, rate);
    }
}

pub struct Adam {
    velocity: Box<NNUEParams>,
    momentum: Box<NNUEParams>,
}

impl Adam {
    const B1: f32 = 0.9;
    const B2: f32 = 0.999;

    fn update(p: &mut f32, m: &mut f32, v: &mut f32, grad: f32, rate: f32) {
        *m = Self::B1 * *m + (1. - Self::B1) * grad;
        *v = Self::B2 * *v + (1. - Self::B2) * grad * grad;
        *p -= rate * *m / (v.sqrt() + 0.000_000_01);
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self {
            velocity: NNUEParams::new(),
            momentum: NNUEParams::new(),
        }
    }
}

impl Optimiser for Adam {
    fn update_feature_weights(
        &mut self,
        params: &mut [f32; INPUT * HIDDEN],
        grads: &[f32; INPUT * HIDDEN],
        adj: f32,
        rate: f32,
    ) {
        for (i, param) in params.iter_mut().enumerate() {
            let grad = adj * grads[i];
            Self::update(
                param,
                &mut self.momentum.feature_weights[i],
                &mut self.velocity.feature_weights[i],
                grad,
                rate,
            );
        }
    }

    fn update_feature_bias(
        &mut self,
        params: &mut [f32; HIDDEN],
        grads: &[f32; HIDDEN],
        adj: f32,
        rate: f32,
    ) {
        for (i, param) in params.iter_mut().enumerate() {
            let grad = adj * grads[i];
            Self::update(
                param,
                &mut self.momentum.feature_bias[i],
                &mut self.velocity.feature_bias[i],
                grad,
                rate,
            );
        }
    }

    fn update_output_weights(
        &mut self,
        params: &mut [f32; 2 * HIDDEN],
        grads: &[f32; 2 * HIDDEN],
        adj: f32,
        rate: f32,
    ) {
        for (i, param) in params.iter_mut().enumerate() {
            let grad = adj * grads[i];
            Self::update(
                param,
                &mut self.momentum.output_weights[i],
                &mut self.velocity.output_weights[i],
                grad,
                rate,
            );
        }
    }

    fn update_output_bias(&mut self, param: &mut f32, mut grad: f32, adj: f32, rate: f32) {
        grad *= adj;
        Self::update(
            param,
            &mut self.momentum.output_bias,
            &mut self.velocity.output_bias,
            grad,
            rate,
        );
    }
}
