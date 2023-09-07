use crate::data::{ChessBoard, DataType};

pub trait InputType {
    type RequiredDataType: DataType;
    const BUCKETS: usize;
    const FACTORISER: bool;

    const SIZE: usize = Self::RequiredDataType::INPUTS * (Self::BUCKETS + Self::FACTORISER as usize);

    fn get_feature_indices(feat: <Self::RequiredDataType as DataType>::FeatureType) -> (usize, usize);
}

pub struct Chess768;
impl InputType for Chess768 {
    type RequiredDataType = ChessBoard;
    const BUCKETS: usize = 1;
    const FACTORISER: bool = false;

    fn get_feature_indices((piece, square, _, _): <Self::RequiredDataType as DataType>::FeatureType) -> (usize, usize) {
        let c = usize::from(piece & 8 > 0);
        let pc = 64 * usize::from(piece & 7);
        let sq = usize::from(square);
        let wfeat = [0, 384][c] + pc + sq;
        let bfeat = [384, 0][c] + pc + (sq ^ 56);
        (wfeat, bfeat)
    }
}

pub struct HalfKA;
impl InputType for HalfKA {
    type RequiredDataType = ChessBoard;
    const BUCKETS: usize = 64;
    const FACTORISER: bool = true;

    fn get_feature_indices((piece, square, our_ksq, opp_ksq): <Self::RequiredDataType as DataType>::FeatureType) -> (usize, usize) {
        let c = usize::from(piece & 8 > 0);
        let pc = 64 * usize::from(piece & 7);
        let sq = usize::from(square);
        let wfeat = 768 * (usize::from(our_ksq) + 1) + [0, 384][c] + pc + sq;
        let bfeat = 768 * (usize::from(opp_ksq) + 1) + [384, 0][c] + pc  + (sq ^ 56);
        (wfeat, bfeat)
    }
}
