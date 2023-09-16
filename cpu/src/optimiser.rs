use crate::NetworkParams;

const B1: f32 = 0.9;
const B2: f32 = 0.999;

pub struct AdamW {
    pub velocity: Box<NetworkParams>,
    pub momentum: Box<NetworkParams>,
    pub decay: f32,
}

impl Default for AdamW {
    fn default() -> Self {
        Self {
            velocity: NetworkParams::new(),
            momentum: NetworkParams::new(),
            decay: 0.01,
        }
    }
}

impl AdamW {
    pub fn update_weights(&mut self, nnue: &mut NetworkParams, grads: &NetworkParams, adj: f32, rate: f32) {
        let decay = 1.0 - self.decay * rate;
        for (i, param) in nnue.iter_mut().enumerate() {
            *param *= decay;
            let grad = adj * grads[i];
            self.momentum[i] = B1 * self.momentum[i] + (1. - B1) * grad;
            self.velocity[i] = B2 * self.velocity[i] + (1. - B2) * grad * grad;
            *param -= rate * self.momentum[i] / (self.velocity[i].sqrt() + 0.000_000_01);
            *param = param.clamp(-1.98, 1.98);
        }
    }
}
