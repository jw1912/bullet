mod accumulator;
pub mod activation;
pub mod inputs;
mod quantise;

pub use accumulator::Accumulator;
pub use inputs::InputType;
pub use quantise::quantise_and_write;

use crate::{
    Activation,
    data::{DataType, Features},
    rng::Rand,
    Data, Input, HIDDEN, util::write_to_bin,
};

pub type NetworkParams = Network<f32>;

pub const NETWORK_SIZE: usize = (Input::SIZE + 3) * HIDDEN + 1;
const FEATURE_BIAS: usize = Input::SIZE * HIDDEN;
const OUTPUT_WEIGHTS: usize = (Input::SIZE + 1) * HIDDEN;
const OUTPUT_BIAS: usize = (Input::SIZE + 3) * HIDDEN;

#[derive(Clone)]
#[repr(C)]
pub struct Network<T> {
    weights: [T; NETWORK_SIZE],
}

impl<T: std::ops::AddAssign<T> + Copy> std::ops::AddAssign<&Network<T>> for Network<T> {
    fn add_assign(&mut self, rhs: &Network<T>) {
        for (i, &j) in self.iter_mut().zip(rhs.iter()) {
            *i += j;
        }
    }
}

impl<T> std::ops::Deref for Network<T> {
    type Target = [T; NETWORK_SIZE];
    fn deref(&self) -> &Self::Target {
        &self.weights
    }
}

impl<T> std::ops::DerefMut for Network<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.weights
    }
}

impl<T> Network<T> {
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

impl NetworkParams {
    pub fn random() -> Box<Self> {
        let mut params = NetworkParams::new();
        let mut gen = Rand::new(173645501);

        for param in params[..FEATURE_BIAS].iter_mut() {
            *param = gen.rand(0.01);
        }

        for param in params[OUTPUT_WEIGHTS..OUTPUT_BIAS].iter_mut() {
            *param = gen.rand(0.01);
        }

        params
    }

    pub fn forward(
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
            activated[0][i] = Activation::activate(accs[0][i]);
            eval += activated[0][i] * self[OUTPUT_WEIGHTS + i];
        }

        for i in 0..HIDDEN {
            activated[1][i] = Activation::activate(accs[1][i]);
            eval += activated[1][i] * self[OUTPUT_WEIGHTS + HIDDEN + i];
        }

        eval
    }

    pub fn backprop(
        &self,
        err: f32,
        grad: &mut NetworkParams,
        accs: &[Accumulator; 2],
        activated: &[[f32; HIDDEN]; 2],
        features: &mut Features,
    ) {
        let mut components = [(0.0, 0.0); HIDDEN];

        for i in 0..HIDDEN {
            components[i] = (
                err * self[OUTPUT_WEIGHTS + i] * Activation::prime(accs[0][i]),
                err * self[OUTPUT_WEIGHTS + HIDDEN + i] * Activation::prime(accs[1][i]),
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

    pub fn write_to_bin(&self, output_path: &str) -> std::io::Result<()> {
        const SIZEOF: usize = std::mem::size_of::<NetworkParams>();
        write_to_bin::<Self, SIZEOF>(self, output_path)
    }
}
