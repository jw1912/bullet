mod accumulator;
pub mod activation;
mod quantise;

pub use accumulator::Accumulator;
pub use activation::Activation;
pub use quantise::QuantisedNNUE;

use crate::position::{Features, Position};

pub const HIDDEN: usize = crate::HIDDEN_SIZE;
pub const INPUT: usize = 768;

pub type NNUEParams = NNUE<f32>;

const NNUE_SIZE: usize = (INPUT + 3) * HIDDEN + 1;
pub const FEATURE_BIAS: usize = INPUT * HIDDEN;
pub const OUTPUT_WEIGHTS: usize = (INPUT + 1) * HIDDEN;
pub const OUTPUT_BIAS: usize = (INPUT + 3) * HIDDEN;

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
#[repr(C)]
pub struct NNUE<T> {
    pub weights: [T; NNUE_SIZE],
}

impl<T: std::ops::AddAssign<T> + Copy> std::ops::AddAssign<&NNUE<T>> for NNUE<T> {
    fn add_assign(&mut self, rhs: &NNUE<T>) {
        for (i, &j) in self.iter_mut().zip(rhs.iter()) {
            *i += j;
        }
    }
}

impl<T> std::ops::Deref for NNUE<T> {
    type Target = [T; NNUE_SIZE];
    fn deref(&self) -> &Self::Target {
        &self.weights
    }
}

impl<T> std::ops::DerefMut for NNUE<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.weights
    }
}

impl<T> NNUE<T> {
    pub fn new() -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }
}

impl NNUEParams {
    pub fn forward<Act: Activation>(
        &self,
        pos: &Position,
        accs: &mut [Accumulator<f32>; 2],
        activated: &mut [[f32; HIDDEN]; 2],
        features: &mut Features,
    ) -> f32 {
        let stm = pos.stm();
        let opp = stm ^ 1;

        for (colour, piece, square) in pos.into_iter() {
            let c = usize::from(colour);
            let pc = 64 * usize::from(piece);
            let sq = usize::from(square);
            let wfeat = [0, 384][c] + pc + sq;
            let bfeat = [384, 0][c] + pc + (sq ^ 56);

            features.push(wfeat, bfeat);
            accs[stm].add_feature(wfeat, self);
            accs[opp].add_feature(bfeat, self);
        }

        let mut eval = self[OUTPUT_BIAS];

        for i in 0..HIDDEN {
            activated[0][i] = Act::activate(accs[0][i]);
            eval += activated[0][i] * self[OUTPUT_WEIGHTS + i];
        }

        for i in 0..HIDDEN {
            activated[1][i] = Act::activate(accs[1][i]);
            eval += activated[1][i] * self[OUTPUT_WEIGHTS + HIDDEN + i];
        }

        eval
    }

    pub fn backprop<Act: Activation>(
        &self,
        err: f32,
        stm: usize,
        grad: &mut NNUEParams,
        accs: &[Accumulator<f32>; 2],
        activated: &[[f32; HIDDEN]; 2],
        features: &mut Features,
    ) {
        let mut components = [(0.0, 0.0); HIDDEN];

        for i in 0..HIDDEN {
            components[i] = (
                err * self[OUTPUT_WEIGHTS + i] * Act::activate_prime(accs[0][i]),
                err * self[OUTPUT_WEIGHTS + HIDDEN + i] * Act::activate_prime(accs[1][i]),
            );

            grad[FEATURE_BIAS + i] += components[i].0 + components[i].1;

            grad[OUTPUT_WEIGHTS + i] += err * activated[0][i];
            grad[OUTPUT_WEIGHTS + HIDDEN + i] += err * activated[1][i];
        }

        let opp = stm ^ 1;

        for (wfeat, bfeat) in features {
            let idxs = [wfeat * HIDDEN, bfeat * HIDDEN];
            let (widx, bidx) = (idxs[stm], idxs[opp]);
            for i in 0..HIDDEN {
                grad[widx + i] += components[i].0;
                grad[bidx + i] += components[i].1;
            }
        }

        grad[OUTPUT_BIAS] += err;
    }
}
