use crate::arch::{NNUEParams, HIDDEN, INPUT};

pub trait Optimiser: Default {
    fn update_feature_weights(
        &mut self,
        params: &mut [f64; INPUT * HIDDEN],
        grads: &[f64; INPUT * HIDDEN],
        adj: f64,
        rate: f64,
    );
    fn update_feature_bias(
        &mut self,
        params: &mut [f64; HIDDEN],
        grads: &[f64; HIDDEN],
        adj: f64,
        rate: f64,
    );
    fn update_output_weights(
        &mut self,
        params: &mut [f64; 2 * HIDDEN],
        grads: &[f64; 2 * HIDDEN],
        adj: f64,
        rate: f64,
    );
    fn update_output_bias(&mut self, param: &mut f64, grad: f64, adj: f64, rate: f64);

    fn update_weights(&mut self, nnue: &mut NNUEParams, grads: &NNUEParams, adj: f64, rate: f64) {
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
    const B1: f64 = 0.9;
    const B2: f64 = 0.999;

    fn update(p: &mut f64, m: &mut f64, v: &mut f64, grad: f64, rate: f64) {
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
        params: &mut [f64; INPUT * HIDDEN],
        grads: &[f64; INPUT * HIDDEN],
        adj: f64,
        rate: f64,
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
        params: &mut [f64; HIDDEN],
        grads: &[f64; HIDDEN],
        adj: f64,
        rate: f64,
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
        params: &mut [f64; 2 * HIDDEN],
        grads: &[f64; 2 * HIDDEN],
        adj: f64,
        rate: f64,
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

    fn update_output_bias(&mut self, param: &mut f64, mut grad: f64, adj: f64, rate: f64) {
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
