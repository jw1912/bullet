use bulletformat::{AtaxxBoard, BulletFormat, ChessBoard};

pub trait InputType {
    type RequiredDataType: BulletFormat + Copy + Send + Sync;
    const BUCKETS: usize;

    const SIZE: usize = Self::RequiredDataType::INPUTS * Self::BUCKETS;

    fn get_feature_indices(
        feat: <Self::RequiredDataType as std::iter::IntoIterator>::Item,
    ) -> (usize, usize);
}

#[derive(Debug)]
pub struct Ataxx147;
impl InputType for Ataxx147 {
    type RequiredDataType = AtaxxBoard;
    const BUCKETS: usize = 1;

    fn get_feature_indices(
        (piece, square): <Self::RequiredDataType as std::iter::IntoIterator>::Item,
    ) -> (usize, usize) {
        let pc = usize::from(piece);
        let sq = usize::from(square);

        let stm_idx = 49 * pc + sq;
        let nstm_idx = if pc == 2 { stm_idx } else { 49 * (pc ^ 1) + sq };

        (stm_idx, nstm_idx)
    }
}

#[derive(Debug)]
pub struct Chess768;
impl InputType for Chess768 {
    type RequiredDataType = ChessBoard;
    const BUCKETS: usize = 1;

    fn get_feature_indices(
        (piece, square, _, _): <Self::RequiredDataType as std::iter::IntoIterator>::Item,
    ) -> (usize, usize) {
        let c = usize::from(piece & 8 > 0);
        let pc = 64 * usize::from(piece & 7);
        let sq = usize::from(square);
        let wfeat = [0, 384][c] + pc + sq;
        let bfeat = [384, 0][c] + pc + (sq ^ 56);
        (wfeat, bfeat)
    }
}

const fn get_num_buckets<const N: usize>(arr: [usize; N]) -> usize {
    let mut idx = 0;
    let mut max = 1;
    while idx < N {
        let val = arr[idx];
        if val > max {
            max = val;
        }
        idx += 1;
    }
    max + 1
}

#[derive(Debug)]
pub struct ChessBuckets;
impl ChessBuckets {
    const BUCKETING: [usize; 64] = crate::BUCKETS;

    const SCALED: [usize; 64] = {
        let mut idx = 0;
        let mut ret = [0; 64];
        while idx < 64 {
            ret[idx] = 768 * Self::BUCKETING[idx];
            idx += 1;
        }
        ret
    };
}

impl InputType for ChessBuckets {
    type RequiredDataType = ChessBoard;
    const BUCKETS: usize = get_num_buckets(Self::BUCKETING);

    fn get_feature_indices(
        (piece, square, our_ksq, opp_ksq): <Self::RequiredDataType as std::iter::IntoIterator>::Item,
    ) -> (usize, usize) {
        let c = usize::from(piece & 8 > 0);
        let pc = 64 * usize::from(piece & 7);
        let sq = usize::from(square);
        let wfeat = Self::SCALED[usize::from(our_ksq)] + [0, 384][c] + pc + sq;
        let bfeat = Self::SCALED[usize::from(opp_ksq)] + [384, 0][c] + pc + (sq ^ 56);
        (wfeat, bfeat)
    }
}

#[derive(Debug)]
pub struct ChessBucketsMirrored;
impl ChessBucketsMirrored {
    const BUCKETING: [usize; 32] = crate::BUCKETS_MIRRORED;

    const SCALED: [usize; 64] = {
        let mut idx = 0;
        let mut ret = [0; 64];
        while idx < 64 {
            let sq = (idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8];
            ret[idx] = 768 * Self::BUCKETING[sq];
            idx += 1;
        }
        ret
    };

    fn get_sq(mut sq: usize, ksq: usize) -> usize {
        if ksq % 8 > 3 {
            sq ^= 7;
        }

        sq
    }
}

impl InputType for ChessBucketsMirrored {
    type RequiredDataType = ChessBoard;
    const BUCKETS: usize = get_num_buckets(Self::BUCKETING);

    fn get_feature_indices(
        (piece, square, our_ksq, opp_ksq): <Self::RequiredDataType as BulletFormat>::FeatureType,
    ) -> (usize, usize) {
        let c = usize::from(piece & 8 > 0);
        let pc = 64 * usize::from(piece & 7);
        let sq = usize::from(square);

        let our_sq = Self::get_sq(sq, usize::from(our_ksq));
        let opp_sq = Self::get_sq(sq, usize::from(opp_ksq)) ^ 56;

        let wfeat = Self::SCALED[usize::from(our_ksq)] + [0, 384][c] + pc + our_sq;
        let bfeat = Self::SCALED[usize::from(opp_ksq)] + [384, 0][c] + pc + opp_sq;
        (wfeat, bfeat)
    }
}
