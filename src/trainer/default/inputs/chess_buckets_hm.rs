use bulletformat::{chess::BoardIter, ChessBoard};

use super::{factorised::Factorises, get_num_buckets, Chess768, Factorised, InputType};

#[derive(Clone, Copy, Debug)]
pub struct ChessBucketsMirrored {
    buckets: [usize; 64],
    num_buckets: usize,
}

impl Default for ChessBucketsMirrored {
    fn default() -> Self {
        Self { buckets: [0; 64], num_buckets: 1 }
    }
}

impl ChessBucketsMirrored {
    pub fn new(buckets: [usize; 32]) -> Self {
        let num_buckets = get_num_buckets(&buckets);
        let buckets = {
            let mut idx = 0;
            let mut ret = [0; 64];
            while idx < 64 {
                let sq = (idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8];
                ret[idx] = 768 * buckets[sq];
                idx += 1;
            }
            ret
        };

        Self { buckets, num_buckets }
    }
}

impl InputType for ChessBucketsMirrored {
    type RequiredDataType = ChessBoard;
    type FeatureIter = ChessBucketsMirroredIter;

    fn max_active_inputs(&self) -> usize {
        32
    }

    fn inputs(&self) -> usize {
        768
    }

    fn buckets(&self) -> usize {
        self.num_buckets
    }

    fn feature_iter(&self, pos: &Self::RequiredDataType) -> Self::FeatureIter {
        let our_ksq = usize::from(pos.our_ksq());
        let opp_ksq = usize::from(pos.opp_ksq());

        ChessBucketsMirroredIter {
            flip: [if our_ksq % 8 > 3 { 7 } else { 0 }, if opp_ksq % 8 > 3 { 7 } else { 0 }],
            buckets: [self.buckets[our_ksq], self.buckets[opp_ksq]],
            board_iter: pos.into_iter(),
        }
    }
}

pub struct ChessBucketsMirroredIter {
    flip: [usize; 2],
    buckets: [usize; 2],
    board_iter: BoardIter,
}

impl Iterator for ChessBucketsMirroredIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.board_iter.next().map(|(piece, square)| {
            let c = usize::from(piece & 8 > 0);
            let pc = 64 * usize::from(piece & 7);
            let sq = usize::from(square);

            let our_sq = sq ^ self.flip[0];
            let opp_sq = sq ^ self.flip[1] ^ 56;

            let wfeat = self.buckets[0] + [0, 384][c] + pc + our_sq;
            let bfeat = self.buckets[1] + [384, 0][c] + pc + opp_sq;
            (wfeat, bfeat)
        })
    }
}

impl Factorises<ChessBucketsMirrored> for Chess768 {
    fn derive_feature(&self, _: &ChessBucketsMirrored, feat: usize) -> usize {
        feat % 768
    }
}

pub type ChessBucketsMirroredFactorised = Factorised<ChessBucketsMirrored, Chess768>;

impl ChessBucketsMirroredFactorised {
    pub fn new(buckets: [usize; 32]) -> Self {
        Self::from_parts(ChessBucketsMirrored::new(buckets), Chess768)
    }
}
