#![allow(unused)]

use bulletformat::ChessBoard;

pub trait InputType {
    const BUCKETS: usize;
    const SIZE: usize;
    const MAX_ACTIVE: usize;

    fn map_features<F: FnMut(usize, usize)>(pos: &ChessBoard, f: F);
}

pub struct Chess768;
impl InputType for Chess768 {
    const BUCKETS: usize = 1;
    const SIZE: usize = 768;
    const MAX_ACTIVE: usize = 32;

    fn map_features<F: FnMut(usize, usize)>(pos: &ChessBoard, mut f: F) {
        for (piece, square) in pos.into_iter() {
            let c = usize::from(piece & 8 > 0);
            let pc = 64 * usize::from(piece & 7);
            let sq = usize::from(square);
            let wfeat = [0, 384][c] + pc + sq;
            let bfeat = [384, 0][c] + pc + (sq ^ 56);
            f(wfeat, bfeat)
        }
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
    const MAX_ACTIVE: usize = 32;

    fn map_features<F: FnMut(usize, usize)>(pos: &ChessBoard, mut f: F) {
        let our_ksq = pos.our_ksq();
        let opp_ksq = pos.opp_ksq();

        for (piece, square) in pos.into_iter() {
            let c = usize::from(piece & 8 > 0);
            let pc = 64 * usize::from(piece & 7);
            let sq = usize::from(square);
    
            let our_sq = Self::get_sq(sq, usize::from(our_ksq));
            let opp_sq = Self::get_sq(sq, usize::from(opp_ksq)) ^ 56;
    
            let wfeat = Self::SCALED[usize::from(our_ksq)] + [0, 384][c] + pc + our_sq;
            let bfeat = Self::SCALED[usize::from(opp_ksq)] + [384, 0][c] + pc + opp_sq;
            f(wfeat, bfeat)
        }
    }
}

pub struct TypoInputs;

impl TypoInputs {
    const FEAT_OFFSET: usize = 768 * Self::BUCKETS;
    const PCKIND_OFFSET: usize = Self::FEAT_OFFSET + 768;
    const OCB_OFFSET: usize = Self::PCKIND_OFFSET + 12;
}

impl InputType for TypoInputs {
    const BUCKETS: usize = get_num_buckets(ChessBucketsMirrored::BUCKETING);
    const SIZE: usize = Self::OCB_OFFSET + 4;
    const MAX_ACTIVE: usize = 32 * 3 + 4;

    fn map_features<F: FnMut(usize, usize)>(pos: &ChessBoard, mut f: F) {
        let our_ksq = pos.our_ksq();
        let opp_ksq = pos.opp_ksq();

        for (piece, square) in pos.into_iter() {
            let c = usize::from(piece & 8 > 0);
            let mut pc = usize::from(piece & 7);
            let sq = usize::from(square);

            // piece kind factoriser - no need for king
            if pc < 6 {
                f(Self::PCKIND_OFFSET + [0, 6][c] + pc, Self::PCKIND_OFFSET + [6, 0][c] + pc);
            }

            // opposite colour bishop
            if pc == 2 {
                let oc = sq & 1;
                f(Self::OCB_OFFSET + [0, 2][c] + oc, Self::OCB_OFFSET + [2, 0][c] + (oc ^ 1));
            }

            pc *= 64;

            // base feature factoriser
            let our_sq = ChessBucketsMirrored::get_sq(sq, usize::from(our_ksq));
            let opp_sq = ChessBucketsMirrored::get_sq(sq, usize::from(opp_ksq)) ^ 56;
            let base_wfeat = [0, 384][c] + pc + our_sq;
            let base_bfeat = [384, 0][c] + pc + opp_sq;
            f(Self::FEAT_OFFSET + base_wfeat, Self::FEAT_OFFSET + base_bfeat);

            // actual king bucketed features
            let wfeat = ChessBucketsMirrored::SCALED[usize::from(our_ksq)] + base_wfeat;
            let bfeat = ChessBucketsMirrored::SCALED[usize::from(opp_ksq)] + base_bfeat;
            f(wfeat, bfeat)
        }
    }
}
