use crate::arch::NNUEParams;

pub trait Optimiser: Default {
    fn update_weights(&mut self, nnue: &mut NNUEParams, grads: &NNUEParams, adj: f32, rate: f32);
}

pub struct Adam {
    velocity: Box<NNUEParams>,
    momentum: Box<NNUEParams>,
}

impl Adam {
    const B1: f32 = 0.9;
    const B2: f32 = 0.999;
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
    fn update_weights(
        &mut self,
        nnue: &mut NNUEParams,
        grads: &NNUEParams,
        adj: f32,
        rate: f32,
    ) {
        for (i, param) in nnue.weights.iter_mut().enumerate() {
            let grad = adj * grads.weights[i];
            self.momentum[i] = Self::B1 * self.momentum[i] + (1. - Self::B1) * grad;
            self.velocity[i] = Self::B2 * self.velocity[i] + (1. - Self::B2) * grad * grad;
            *param -= rate * self.momentum[i] / (self.velocity[i].sqrt() + 0.000_000_01);
        }
    }
}

pub struct AdamW {
    velocity: Box<NNUEParams>,
    momentum: Box<NNUEParams>,
    decay: f32,
}

impl AdamW {
    const B1: f32 = 0.9;
    const B2: f32 = 0.999;
}

impl Default for AdamW {
    fn default() -> Self {
        Self {
            velocity: NNUEParams::new(),
            momentum: NNUEParams::new(),
            decay: 0.01,
        }
    }
}

impl Optimiser for AdamW {
    fn update_weights(
        &mut self,
        nnue: &mut NNUEParams,
        grads: &NNUEParams,
        adj: f32,
        rate: f32,
    ) {
        let decay = 1.0 - self.decay * rate;
        for (i, param) in nnue.weights.iter_mut().enumerate() {
            let grad = adj * grads.weights[i];
            self.momentum[i] = Self::B1 * self.momentum[i] + (1. - Self::B1) * grad;
            self.velocity[i] = Self::B2 * self.velocity[i] + (1. - Self::B2) * grad * grad;
            *param *= decay;
            *param -= rate * self.momentum[i] / (self.velocity[i].sqrt() + 0.000_000_01);
        }
    }
}
