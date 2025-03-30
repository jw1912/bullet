use bulletformat::ChessBoard;

use super::{get_num_buckets, Chess768, Factorises, SparseInputType};

#[derive(Clone, Copy, Debug)]
pub struct ChessBuckets {
    buckets: [usize; 64],
    num_buckets: usize,
}

impl ChessBuckets {
    pub fn new(buckets: [usize; 64]) -> Self {
        Self { buckets, num_buckets: get_num_buckets(&buckets) }
    }
}

impl SparseInputType for ChessBuckets {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        768 * self.num_buckets
    }

    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let our_bucket = 768 * self.buckets[usize::from(pos.our_ksq())];
        let opp_bucket = 768 * self.buckets[usize::from(pos.opp_ksq())];

        Chess768.map_features(pos, |stm, ntm| f(our_bucket + stm, opp_bucket + ntm));
    }

    fn shorthand(&self) -> String {
        format!("768x{}", self.num_buckets)
    }

    fn description(&self) -> String {
        "King bucketed psqt chess inputs".to_string()
    }
}

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

        let mut expanded = [0; 64];
        for (idx, elem) in expanded.iter_mut().enumerate() {
            *elem = buckets[(idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8]];
        }

        Self { buckets: expanded, num_buckets }
    }
}

impl SparseInputType for ChessBucketsMirrored {
    type RequiredDataType = ChessBoard;

    /// The total number of inputs
    fn num_inputs(&self) -> usize {
        768 * self.num_buckets
    }

    /// The maximum number of active inputs
    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let get = |ksq| (if ksq % 8 > 3 { 7 } else { 0 }, 768 * self.buckets[usize::from(ksq)]);
        let (stm_flip, stm_bucket) = get(pos.our_ksq());
        let (ntm_flip, ntm_bucket) = get(pos.opp_ksq());

        Chess768.map_features(pos, |stm, ntm| f(stm_bucket + (stm ^ stm_flip), ntm_bucket + (ntm ^ ntm_flip)));
    }

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String {
        format!("768x{}hm", self.num_buckets)
    }

    /// Description of the input type
    fn description(&self) -> String {
        "Horizontally mirrored, king bucketed psqt chess inputs".to_string()
    }
}

impl Factorises<ChessBuckets> for Chess768 {
    fn derive_feature(&self, _: &ChessBuckets, feat: usize) -> Option<usize> {
        Some(feat % 768)
    }
}

impl Factorises<ChessBucketsMirrored> for Chess768 {
    fn derive_feature(&self, _: &ChessBucketsMirrored, feat: usize) -> Option<usize> {
        Some(feat % 768)
    }
}
