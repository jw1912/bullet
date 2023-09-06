mod chess768;

pub use chess768::Chess768;

use crate::{position::Features, network::{Accumulator, NNUEParams}};

pub trait InputType {
    type FeatureType;
    const SIZE: usize;

    fn update_features_and_accumulator(
        feature: Self::FeatureType,
        stm: usize,
        features: &mut Features,
        accs: &mut [Accumulator; 2],
        nnue: &NNUEParams,
    );
}