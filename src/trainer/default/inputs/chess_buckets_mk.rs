use bulletformat::{chess::BoardIter, ChessBoard};

use super::{get_num_buckets, InputType};

#[derive(Clone, Copy, Debug)]
pub struct ChessBucketsMergedKings {
    buckets: [usize; 64],
    num_buckets: usize,
}

impl ChessBucketsMergedKings {
    pub fn new(buckets: [usize; 64]) -> Self {
        let num_buckets = get_num_buckets(&buckets);
        let buckets = {
            let mut idx = 0;
            let mut ret = [0; 64];
            while idx < 64 {
                ret[idx] = 704 * buckets[idx];
                idx += 1;
            }
            ret
        };

        Self { buckets, num_buckets }
    }
}

impl InputType for ChessBucketsMergedKings {
    type RequiredDataType = ChessBoard;
    type FeatureIter = ChessBucketsMergedKingsIter;

    fn max_active_inputs(&self) -> usize {
        32
    }

    fn inputs(&self) -> usize {
        704
    }

    fn buckets(&self) -> usize {
        self.num_buckets
    }

    fn feature_iter(&self, pos: &Self::RequiredDataType) -> Self::FeatureIter {
        let buckets = [self.buckets[usize::from(pos.our_ksq())], self.buckets[usize::from(pos.opp_ksq())]];

        ChessBucketsMergedKingsIter { buckets, board_iter: pos.into_iter() }
    }

    fn description(&self) -> String {
        "King bucketed psqt chess inputs, with merged kings".to_string()
    }
}

pub struct ChessBucketsMergedKingsIter {
    buckets: [usize; 2],
    board_iter: BoardIter,
}

impl Iterator for ChessBucketsMergedKingsIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.board_iter.next().map(|(piece, square)| {
            let c = usize::from(piece & 8 > 0 && piece & 7 != 5);
            let pc = 64 * usize::from(piece & 7);
            let sq = usize::from(square);
            let wfeat = self.buckets[0] + [0, 384][c] + pc + sq;
            let bfeat = self.buckets[1] + [384, 0][c] + pc + (sq ^ 56);
            (wfeat, bfeat)
        })
    }
}
