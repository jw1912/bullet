use bulletformat::{chess::BoardIter, ChessBoard};

use super::{get_num_buckets, InputType};

#[derive(Clone, Copy, Debug)]
pub struct ChessBuckets {
    buckets: [usize; 64],
    num_buckets: usize,
}

impl Default for ChessBuckets {
    fn default() -> Self {
        Self { buckets: [0; 64], num_buckets: 1 }
    }
}

impl ChessBuckets {
    pub fn new(buckets: [usize; 64]) -> Self {
        let num_buckets = get_num_buckets(&buckets);
        let buckets = {
            let mut idx = 0;
            let mut ret = [0; 64];
            while idx < 64 {
                ret[idx] = 768 * buckets[idx];
                idx += 1;
            }
            ret
        };

        Self { buckets, num_buckets }
    }
}

impl InputType for ChessBuckets {
    type RequiredDataType = ChessBoard;
    type FeatureIter = ChessBucketsIter;

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
        let buckets = [self.buckets[usize::from(pos.our_ksq())], self.buckets[usize::from(pos.opp_ksq())]];

        ChessBucketsIter { buckets, board_iter: pos.into_iter() }
    }
}

pub struct ChessBucketsIter {
    buckets: [usize; 2],
    board_iter: BoardIter,
}

impl Iterator for ChessBucketsIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.board_iter.next().map(|(piece, square)| {
            let c = usize::from(piece & 8 > 0);
            let pc = 64 * usize::from(piece & 7);
            let sq = usize::from(square);
            let wfeat = self.buckets[0] + [0, 384][c] + pc + sq;
            let bfeat = self.buckets[1] + [384, 0][c] + pc + (sq ^ 56);
            (wfeat, bfeat)
        })
    }
}
