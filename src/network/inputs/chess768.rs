use crate::{position::Features, network::{Accumulator, NNUEParams}};
use super::InputType;

pub struct Chess768;

impl InputType for Chess768 {
    type FeatureType = (u8, u8, u8);
    const SIZE: usize = 768;

    fn update_features_and_accumulator(
        (colour, piece, square): Self::FeatureType,
        stm: usize,
        features: &mut Features,
        accs: &mut [Accumulator; 2],
        nnue: &NNUEParams
    ) {
        let c = usize::from(colour);
        let pc = 64 * usize::from(piece);
        let sq = usize::from(square);
        let wfeat = [0, 384][c] + pc + sq;
        let bfeat = [384, 0][c] + pc + (sq ^ 56);

        features.push(wfeat, bfeat);
        accs[stm].add_feature(wfeat, nnue);
        accs[stm ^ 1].add_feature(bfeat, nnue);
    }
}