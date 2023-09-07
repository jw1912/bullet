use crate::{Data, data::Features, network::{Accumulator, NNUEParams}};

pub trait InputType {
    type DataType;
    const SIZE: usize;

    fn update_features_and_accumulator(
        pos: &Self::DataType,
        features: &mut Features,
        accs: &mut [Accumulator; 2],
        nnue: &NNUEParams,
    );
}

pub struct Chess768;

impl InputType for Chess768 {
    type DataType = Data;
    const SIZE: usize = 768;

    #[inline]
    fn update_features_and_accumulator(
        pos: &Self::DataType,
        features: &mut Features,
        accs: &mut [Accumulator; 2],
        nnue: &NNUEParams
    ) {
        for (colour, piece, square) in pos.into_iter() {
            let c = usize::from(colour);
            let pc = 64 * usize::from(piece);
            let sq = usize::from(square);
            let wfeat = [0, 384][c] + pc + sq;
            let bfeat = [384, 0][c] + pc + (sq ^ 56);

            features.push(wfeat, bfeat);
            accs[0].add_feature(wfeat, nnue);
            accs[1].add_feature(bfeat, nnue);
        }
    }
}