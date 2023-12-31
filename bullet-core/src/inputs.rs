use bulletformat::{AtaxxBoard, BulletFormat, ChessBoard};

pub trait InputType: Send + Sync + Copy + Default {
    type RequiredDataType: BulletFormat + Copy + Send + Sync;

    fn inputs(&self) -> usize;

    fn buckets(&self) -> usize;

    fn size(&self) -> usize {
        self.inputs() * self.buckets()
    }

    fn get_feature_indices(
        &self,
        feat: <Self::RequiredDataType as std::iter::IntoIterator>::Item,
    ) -> (usize, usize);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Ataxx147;
impl InputType for Ataxx147 {
    type RequiredDataType = AtaxxBoard;

    fn inputs(&self) -> usize {
        147
    }

    fn buckets(&self) -> usize {
        1
    }

    fn get_feature_indices(
        &self,
        (piece, square): <Self::RequiredDataType as std::iter::IntoIterator>::Item,
    ) -> (usize, usize) {
        let pc = usize::from(piece);
        let sq = usize::from(square);

        let stm_idx = 49 * pc + sq;
        let nstm_idx = if pc == 2 { stm_idx } else { 49 * (pc ^ 1) + sq };

        (stm_idx, nstm_idx)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Chess768;
impl InputType for Chess768 {
    type RequiredDataType = ChessBoard;

    fn inputs(&self) -> usize {
        768
    }

    fn buckets(&self) -> usize {
        1
    }

    fn get_feature_indices(
        &self,
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

fn get_num_buckets<const N: usize>(arr: &[usize; N]) -> usize {
    let mut max = 0;
    for &val in arr {
        max = max.max(val)
    }
    max + 1
}

#[derive(Clone, Copy, Debug)]
pub struct ChessBuckets {
    buckets: [usize; 64],
    num_buckets: usize,
}

impl Default for ChessBuckets {
    fn default() -> Self {
        Self {
            buckets: [0; 64],
            num_buckets: 1,
        }
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

        Self {
            buckets,
            num_buckets,
        }
    }
}

impl InputType for ChessBuckets {
    type RequiredDataType = ChessBoard;

    fn inputs(&self) -> usize {
        768
    }

    fn buckets(&self) -> usize {
        self.num_buckets
    }

    fn get_feature_indices(
        &self,
        (piece, square, our_ksq, opp_ksq): <Self::RequiredDataType as std::iter::IntoIterator>::Item,
    ) -> (usize, usize) {
        let c = usize::from(piece & 8 > 0);
        let pc = 64 * usize::from(piece & 7);
        let sq = usize::from(square);
        let wfeat = self.buckets[usize::from(our_ksq)] + [0, 384][c] + pc + sq;
        let bfeat = self.buckets[usize::from(opp_ksq)] + [384, 0][c] + pc + (sq ^ 56);
        (wfeat, bfeat)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ChessBucketsMirrored {
    buckets: [usize; 64],
    num_buckets: usize,
}

impl Default for ChessBucketsMirrored {
    fn default() -> Self {
        Self {
            buckets: [0; 64],
            num_buckets: 1,
        }
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

        Self {
            buckets,
            num_buckets,
        }
    }

    fn get_sq(mut sq: usize, ksq: usize) -> usize {
        if ksq % 8 > 3 {
            sq ^= 7;
        }

        sq
    }
}

impl InputType for ChessBucketsMirrored {
    type RequiredDataType = ChessBoard;

    fn inputs(&self) -> usize {
        768
    }

    fn buckets(&self) -> usize {
        self.num_buckets
    }

    fn get_feature_indices(
        &self,
        (piece, square, our_ksq, opp_ksq): <Self::RequiredDataType as BulletFormat>::FeatureType,
    ) -> (usize, usize) {
        let c = usize::from(piece & 8 > 0);
        let pc = 64 * usize::from(piece & 7);
        let sq = usize::from(square);

        let our_sq = Self::get_sq(sq, usize::from(our_ksq));
        let opp_sq = Self::get_sq(sq, usize::from(opp_ksq)) ^ 56;

        let wfeat = self.buckets[usize::from(our_ksq)] + [0, 384][c] + pc + our_sq;
        let bfeat = self.buckets[usize::from(opp_ksq)] + [384, 0][c] + pc + opp_sq;
        (wfeat, bfeat)
    }
}
