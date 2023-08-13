use crate::position::Position;
use std::ops::{AddAssign, Index, IndexMut};

pub const INPUT: usize = 768;
pub const HIDDEN: usize = 16;
pub const K: f64 = 3.6;

fn activate(x: f64) -> f64 {
    x.max(0.0)
}

fn activate_prime(x: f64) -> f64 {
    if x < 0.0 {0.0} else {1.0}
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

impl NNUEParams {
    pub fn test_eval(&self) {
        println!("{:?}", self.output_weights);
        println!("{:?}", self.output_bias);
        println!("{:?}", self.feature_bias);
        const FENS: [&str; 3] = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - ",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
        ];
        for fen in FENS {
            let pos = fen.parse::<Position>().unwrap();
            let score = self.eval(&pos);
            println!("eval: {score}");
        }
    }

    fn eval(&self, pos: &Position) -> f64 {
        let mut acc = Accumulator::new(self.feature_bias);

        for &feature in pos.active.iter().take(pos.num) {
            acc.add_feature(usize::from(feature), self);
        }

        let mut eval = self.output_bias;
        for (&i, &w) in acc.0.iter().zip(&self.output_weights) {
            eval += activate(i) * w;
        }

        eval * 400.
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

    let mut eval = nnue.output_bias;
    let mut act = Accumulator::new([0.0; HIDDEN]);
    for (idx, (&i, &w)) in acc.0.iter().zip(&nnue.output_weights).enumerate() {
        act[idx] = activate(i);
        eval += act[idx] * w;
    }

    let sigmoid = 1. / (1. + (-eval * K).exp());
    let err = (sigmoid - pos.result) * sigmoid * (1. - sigmoid);
    *error += (sigmoid - pos.result).powi(2);

    // gradient calculation
    for i in 0..HIDDEN {
        let component = err * nnue.output_weights[i] * activate_prime(acc[i]);

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
