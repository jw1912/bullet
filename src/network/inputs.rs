pub trait InputType {
    type FeatureType;
    const SIZE: usize;
    const FACTORISER: bool;

    fn get_feature_indices(feat: Self::FeatureType) -> (usize, usize);
}

pub struct Chess768;
impl InputType for Chess768 {
    type FeatureType = (u8, u8, u8, u8);
    const SIZE: usize = 768;
    const FACTORISER: bool = false;

    fn get_feature_indices((piece, square, _, _): Self::FeatureType) -> (usize, usize) {
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
    type FeatureType = (u8, u8, u8, u8);
    const SIZE: usize = 65 * 768;
    const FACTORISER: bool = true;

    fn get_feature_indices((piece, square, our_ksq, opp_ksq): Self::FeatureType) -> (usize, usize) {
        let c = usize::from(piece & 8 > 0);
        let pc = 64 * usize::from(piece & 7);
        let sq = usize::from(square);
        let wfeat = 768 * (usize::from(our_ksq) + 1) + [0, 384][c] + pc + sq;
        let bfeat = 768 * (usize::from(opp_ksq) + 1) + [384, 0][c] + pc  + (sq ^ 56);
        (wfeat, bfeat)
    }
}

