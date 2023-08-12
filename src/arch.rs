use crate::position::Position;

const INPUT: usize = 768;
const HIDDEN: usize = 32;

const CR_MIN: f64 = 0.0;
const CR_MAX: f64 = 255.0;

fn activate(x: f64) -> f64 {
    x.clamp(CR_MIN, CR_MAX)
}

fn activate_prime(x: f64) -> f64 {
    if x <= CR_MIN || x >= CR_MAX {0.0} else {1.0}
}

#[repr(C)]
pub struct NNUEParams {
    feature_weights: [f64; INPUT * HIDDEN],
    feature_bias: [f64; HIDDEN],
    output_weights: [f64; HIDDEN],
    output_bias: f64,
}

impl Default for NNUEParams {
    fn default() -> Self {
        Self {
            feature_weights: [0.0; INPUT * HIDDEN],
            feature_bias: [0.0; HIDDEN],
            output_weights: [0.0; HIDDEN],
            output_bias: 0.0,
        }
    }
}

#[derive(Clone, Copy)]
struct Accumulator<const SIZE: usize>([f64; SIZE]);

impl<const SIZE: usize> std::ops::Index<usize> for Accumulator<SIZE> {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const SIZE: usize> std::ops::IndexMut<usize> for Accumulator<SIZE> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const SIZE: usize> Accumulator<SIZE> {
    fn new(vals: [f64; SIZE]) -> Self {
        Self(vals)
    }

    fn add_feature(&mut self, feature_idx: usize, nnue: &NNUEParams) {
        let start = feature_idx * SIZE;
        for (i, d) in self.0.iter_mut().zip(&nnue.feature_weights[start..start + SIZE]) {
            *i += *d;
        }
    }
}

pub fn eval(pos: &Position, nnue: &NNUEParams) -> f64 {
    let mut acc = Accumulator::new(nnue.feature_bias);

    for &feature in pos.active.iter().take(pos.num) {
        acc.add_feature(usize::from(feature), nnue);
    }

    let mut sum = 0.0;
    for (&i, &w) in acc.0.iter().zip(&nnue.output_weights) {
        sum += activate(i) * w;
    }

    sum + nnue.output_bias
}

pub fn update_single_grad(pos: &Position, nnue: &NNUEParams, grad: &mut NNUEParams) {
    // eval and helper calculations

    let mut acc = Accumulator::new(nnue.feature_bias);

    for &feature in pos.active.iter().take(pos.num) {
        acc.add_feature(usize::from(feature), nnue);
    }

    let mut sum = 0.0;
    let mut act = Accumulator::new([0.0; HIDDEN]);
    for (idx, (&i, &w)) in acc.0.iter().zip(&nnue.output_weights).enumerate() {
        act[idx] = activate(i);
        sum += act[idx] * w;
    }

    // we might be using these a lot
    let mut act_prime = Accumulator::new([0.0; HIDDEN]);
    for (idx, &val) in acc.0.iter().enumerate() {
        act_prime[idx] = activate_prime(val);
    }

    let eval = sum + nnue.output_bias;

    // gradient calculation

    let err = eval - pos.result;
    for i in 0..HIDDEN {
        let component = err * nnue.output_weights[i] * act_prime[i];

        // update feature weight gradients
        for &j in pos.active.iter().take(pos.num) {
            grad.feature_weights[usize::from(j) * HIDDEN + i] += component;
        }

        // update feature bias gradients
        grad.feature_bias[i] += component;

        // update output weight gradients
        grad.output_weights[i] += err * act[i];
    }

    // update output bias gradient
    grad.output_bias += err;
}