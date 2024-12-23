pub trait InputType {
    const BUCKETS: usize;
    const SIZE: usize;

    fn get_feature_indices(psq: (u8, u8), kings: (u8, u8)) -> (usize, usize);
}

pub struct Chess768;
impl InputType for Chess768 {
    const BUCKETS: usize = 1;
    const SIZE: usize = 768;

    fn get_feature_indices((piece, square): (u8, u8), _: (u8, u8)) -> (usize, usize) {
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
    let mut max = 0;
    while idx < N {
        let val = arr[idx];
        if val > max {
            max = val;
        }
        idx += 1;
    }
    max + 1
}

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
    const BUCKETS: usize = get_num_buckets(Self::BUCKETING);
    const SIZE: usize = 768 * Self::BUCKETS;

    fn get_feature_indices(
        (piece, square): (u8, u8), (our_ksq, opp_ksq): (u8, u8)
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
