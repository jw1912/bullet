use crate::position::Position;
use std::ops::{AddAssign, Index, IndexMut};

pub const INPUT: usize = 768;
pub const HIDDEN: usize = 16;

const CR_MIN: f64 = 0.0;
const CR_MAX: f64 = 1.0;

fn activate(x: f64) -> f64 {
    x.clamp(CR_MIN, CR_MAX)
}

fn activate_prime(x: f64) -> f64 {
    if x <= CR_MIN || x >= CR_MAX {0.0} else {1.0}
}

pub type NNUEParams = NNUE<f64>;

#[derive(Clone)]
#[repr(C)]
pub struct NNUE<T> {
    pub feature_weights: [T; INPUT * HIDDEN],
    pub feature_bias: [T; HIDDEN],
    pub output_weights: [T; HIDDEN],
    pub output_bias: T,
}

impl<T: Copy + Default> Default for NNUE<T> {
    fn default() -> Self {
        Self {
            feature_weights: [T::default(); INPUT * HIDDEN],
            feature_bias: [T::default(); HIDDEN],
            output_weights: [T::default(); HIDDEN],
            output_bias: T::default(),
        }
    }
}

impl<T: AddAssign<T> + Copy> AddAssign<NNUE<T>> for NNUE<T> {
    fn add_assign(&mut self, rhs: NNUE<T>) {
        for (i, &j) in self.feature_weights.iter_mut().zip(rhs.feature_weights.iter()) {
            *i += j
        }

        for (i, &j) in self.output_weights.iter_mut().zip(rhs.output_weights.iter()) {
            *i += j
        }

        for (i, &j) in self.feature_bias.iter_mut().zip(rhs.feature_bias.iter()) {
            *i += j
        }

        self.output_bias += rhs.output_bias;
    }
}

#[derive(Clone, Copy)]
pub struct Accumulator<T, const SIZE: usize>(pub [T; SIZE]);

impl<T, const SIZE: usize> Index<usize> for Accumulator<T, SIZE> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, const SIZE: usize> IndexMut<usize> for Accumulator<T, SIZE> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T: Copy + AddAssign<T>, const SIZE: usize> Accumulator<T, SIZE> {
    pub fn new(vals: [T; SIZE]) -> Self {
        Self(vals)
    }

    pub fn add_feature(&mut self, feature_idx: usize, nnue: &NNUE<T>) {
        let start = feature_idx * SIZE;
        for (i, d) in self.0.iter_mut().zip(&nnue.feature_weights[start..start + SIZE]) {
            *i += *d;
        }
    }
}

pub fn update_single_grad(pos: &Position, nnue: &NNUEParams, grad: &mut NNUEParams, error: &mut f64) {
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
    *error += err.powi(2);
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
