use bulletformat::{chess::BoardIter, ChessBoard};

use super::{get_num_buckets, InputType};

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

#[derive(Clone, Copy, Debug)]
pub struct ChessBucketsMirroredFactorised {
    buckets: [usize; 64],
    num_buckets: usize,
}

impl Default for ChessBucketsMirroredFactorised {
    fn default() -> Self {
        Self { buckets: [0; 64], num_buckets: 1 }
    }
}

impl ChessBucketsMirroredFactorised {
    pub fn new(buckets: [usize; 32]) -> Self {
        let num_buckets = get_num_buckets(&buckets);
        let buckets = {
            let mut idx = 0;
            let mut ret = [0; 64];
            while idx < 64 {
                let sq = (idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8];
                ret[idx] = 768 * (1 + buckets[sq]);
                idx += 1;
            }
            ret
        };

        Self { buckets, num_buckets }
    }
}

impl InputType for ChessBucketsMirroredFactorised {
    type RequiredDataType = ChessBoard;
    type FeatureIter = ChessBucketsMirroredFactorisedIter;

    fn max_active_inputs(&self) -> usize {
        64
    }

    fn inputs(&self) -> usize {
        768
    }

    fn buckets(&self) -> usize {
        self.num_buckets + 1
    }

    fn feature_iter(&self, pos: &Self::RequiredDataType) -> Self::FeatureIter {
        let our_ksq = usize::from(pos.our_ksq());
        let opp_ksq = usize::from(pos.opp_ksq());

        ChessBucketsMirroredFactorisedIter {
            flip: [if our_ksq % 8 > 3 { 7 } else { 0 }, if opp_ksq % 8 > 3 { 7 } else { 0 }],
            buckets: [self.buckets[our_ksq], self.buckets[opp_ksq]],
            board_iter: pos.into_iter(),
            queued: None,
        }
    }
}

pub struct ChessBucketsMirroredFactorisedIter {
    flip: [usize; 2],
    buckets: [usize; 2],
    board_iter: BoardIter,
    queued: Option<(usize, usize)>,
}

impl Iterator for ChessBucketsMirroredFactorisedIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(feats) = self.queued {
            self.queued = None;
            Some(feats)
        } else {
            self.board_iter.next().map(|(piece, square)| {
                let c = usize::from(piece & 8 > 0);
                let pc = 64 * usize::from(piece & 7);
                let sq = usize::from(square);

                let our_sq = sq ^ self.flip[0];
                let opp_sq = sq ^ self.flip[1] ^ 56;

                let wfeat = self.buckets[0] + [0, 384][c] + pc + our_sq;
                let bfeat = self.buckets[1] + [384, 0][c] + pc + opp_sq;
                self.queued = Some((wfeat % 768, bfeat % 768));
                (wfeat, bfeat)
            })
        }
    }
}
