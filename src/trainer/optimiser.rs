use crate::network::NNUEParams;

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

    fn update_single(
        &mut self,
        i: usize,
        param: &mut f32,
        grads: &NNUEParams,
        adj: f32,
        rate: f32,
    ) {
        let grad = adj * grads[i];
        self.momentum[i] = Self::B1 * self.momentum[i] + (1. - Self::B1) * grad;
        self.velocity[i] = Self::B2 * self.velocity[i] + (1. - Self::B2) * grad * grad;
        *param -= rate * self.momentum[i] / (self.velocity[i].sqrt() + 0.000_000_01);
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
    fn update_weights(&mut self, nnue: &mut NNUEParams, grads: &NNUEParams, adj: f32, rate: f32) {
        for (i, param) in nnue.iter_mut().enumerate() {
            self.update_single(i, param, grads, adj, rate);
            *param = param.clamp(-1.98, 1.98);
        }
    }
}

pub struct AdamW {
    adam: Adam,
    decay: f32,
}

impl Default for AdamW {
    fn default() -> Self {
        Self {
            adam: Adam::default(),
            decay: 0.01,
        }
    }
}

impl Optimiser for AdamW {
    fn update_weights(&mut self, nnue: &mut NNUEParams, grads: &NNUEParams, adj: f32, rate: f32) {
        let decay = 1.0 - self.decay * rate;
        for (i, param) in nnue.iter_mut().enumerate() {
            *param *= decay;
            self.adam.update_single(i, param, grads, adj, rate);
            *param = param.clamp(-1.98, 1.98);
        }
    }
}
