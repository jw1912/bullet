pub trait InputType {
    type FeatureType;
    const SIZE: usize;

    fn get_feature_indices(feat: Self::FeatureType) -> (usize, usize);
}

pub struct Chess768;
impl InputType for Chess768 {
    type FeatureType = (u8, u8, u8, u8);
    const SIZE: usize = 768;

    fn get_feature_indices((colour, piece, square, _): Self::FeatureType) -> (usize, usize) {
        let c = usize::from(colour);
        let pc = 64 * usize::from(piece);
        let sq = usize::from(square);
        let wfeat = [0, 384][c] + pc + sq;
        let bfeat = [384, 0][c] + pc + (sq ^ 56);
        (wfeat, bfeat)
    }
}

pub struct HalfKA;
impl InputType for HalfKA {
    type FeatureType = (u8, u8, u8, u8);
    const SIZE: usize = 64 * 768;

    fn get_feature_indices((colour, piece, square, ksq): Self::FeatureType) -> (usize, usize) {
        let bucket = 768 * usize::from(ksq);
        let c = usize::from(colour);
        let pc = 64 * usize::from(piece);
        let sq = usize::from(square);
        let wfeat = bucket + [0, 384][c] + pc + sq;
        let bfeat = bucket + [384, 0][c] + pc + (sq ^ 56);
        (wfeat, bfeat)
    }
}

