use bulletformat::{chess::BoardIter, ChessBoard};

use super::{factorised::Factorises, get_num_buckets, Chess768, Factorised, InputType};

#[derive(Clone, Copy, Debug)]
pub struct ChessBucketsMergedKingsMirrored {
    buckets: [usize; 64],
    num_buckets: usize,
}

impl Default for ChessBucketsMergedKingsMirrored {
    #[rustfmt::skip]
    fn default() -> Self {
        Self::new([
             0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
            12, 13, 14, 15,
            16, 17, 18, 19,
            20, 21, 22, 23,
            24, 25, 26, 27,
            28, 29, 30, 31,
        ])
    }
}

impl ChessBucketsMergedKingsMirrored {
    pub fn new(buckets: [usize; 32]) -> Self {
        let num_buckets = get_num_buckets(&buckets);
        let buckets = {
            let mut idx = 0;
            let mut ret = [0; 64];
            while idx < 64 {
                let sq = (idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8];
                ret[idx] = 704 * buckets[sq];
                idx += 1;
            }
            ret
        };

        Self { buckets, num_buckets }
    }
}

impl InputType for ChessBucketsMergedKingsMirrored {
    type RequiredDataType = ChessBoard;
    type FeatureIter = ChessBucketsMergedKingsMirroredIter;

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
        let our_ksq = usize::from(pos.our_ksq());
        let opp_ksq = usize::from(pos.opp_ksq());

        ChessBucketsMergedKingsMirroredIter {
            flip: [if our_ksq % 8 > 3 { 7 } else { 0 }, if opp_ksq % 8 > 3 { 7 } else { 0 }],
            buckets: [self.buckets[our_ksq], self.buckets[opp_ksq]],
            board_iter: pos.into_iter(),
        }
    }

    fn description(&self) -> String {
        "Horizontally mirrored, king bucketed psqt chess inputs, with merged kings".to_string()
    }
}

pub struct ChessBucketsMergedKingsMirroredIter {
    flip: [usize; 2],
    buckets: [usize; 2],
    board_iter: BoardIter,
}

impl Iterator for ChessBucketsMergedKingsMirroredIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.board_iter.next().map(|(piece, square)| {
            let c = usize::from(piece & 8 > 0 && piece & 7 != 5);
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

impl Factorises<ChessBucketsMergedKingsMirrored> for Chess768 {
    fn derive_feature(&self, _: &ChessBucketsMergedKingsMirrored, _feat: usize) -> Option<usize> {
        todo!();
    }
}

pub type ChessBucketsMergedKingsMirroredFactorised = Factorised<ChessBucketsMergedKingsMirrored, Chess768>;

impl ChessBucketsMergedKingsMirroredFactorised {
    pub fn new(buckets: [usize; 32]) -> Self {
        Self::from_parts(ChessBucketsMergedKingsMirrored::new(buckets), Chess768)
    }
}
