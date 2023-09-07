mod accumulator;
pub mod activation;
pub mod inputs;
mod quantise;

pub use accumulator::Accumulator;
pub use activation::Activation;
pub use inputs::InputType;
pub use quantise::QuantisedNNUE;

use crate::{data::Features, rng::Rand, Data, Input, HIDDEN};

pub type NNUEParams = NNUE<f32>;

const NNUE_SIZE: usize = (Input::SIZE + 3) * HIDDEN + 1;
const FEATURE_BIAS: usize = Input::SIZE * HIDDEN;
const OUTPUT_WEIGHTS: usize = (Input::SIZE + 1) * HIDDEN;
const OUTPUT_BIAS: usize = (Input::SIZE + 3) * HIDDEN;

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
#[repr(C)]
pub struct NNUE<T> {
    weights: [T; NNUE_SIZE],
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
    pub fn random() -> Box<Self> {
        let mut params = NNUEParams::new();
        let mut gen = Rand::new(173645501);

        for param in params[..FEATURE_BIAS].iter_mut() {
            *param = gen.rand(0.01);
        }

        for param in params[OUTPUT_WEIGHTS..OUTPUT_BIAS].iter_mut() {
            *param = gen.rand(0.01);
        }

        params
    }

    pub fn forward<Act: Activation>(
        &self,
        pos: &Data,
        accs: &mut [Accumulator; 2],
        activated: &mut [[f32; HIDDEN]; 2],
        features: &mut Features,
    ) -> f32 {
        for feat in pos.into_iter() {
            let (wfeat, bfeat) = Input::get_feature_indices(feat);

            features.push(wfeat, bfeat);
            accs[0].add_feature(wfeat, self);
            accs[1].add_feature(bfeat, self);
            if Input::FACTORISER {
                accs[0].add_feature(wfeat % Data::INPUTS, self);
                accs[1].add_feature(bfeat % Data::INPUTS, self);
            }
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
        grad: &mut NNUEParams,
        accs: &[Accumulator; 2],
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

        for (wfeat, bfeat) in features {
            let (widx, bidx) = (wfeat * HIDDEN, bfeat * HIDDEN);
            for i in 0..HIDDEN {
                grad[widx + i] += components[i].0;
                grad[bidx + i] += components[i].1;
            }
        }

        grad[OUTPUT_BIAS] += err;
    }
}
