mod ataxx147;
mod chess768;
mod chess_buckets;
mod chess_buckets_mk;
mod factorised;

#[allow(deprecated)]
mod legacy;

use super::loader::LoadableDataType;

pub use ataxx147::{Ataxx147, Ataxx98};
pub use chess768::Chess768;
pub use chess_buckets::{ChessBuckets, ChessBucketsMirrored};
pub use chess_buckets_mk::{ChessBucketsMergedKings, ChessBucketsMergedKingsMirrored};
pub use factorised::{Factorised, Factorises};

#[allow(deprecated)]
pub use legacy::InputType;

pub type ChessBucketsFactorised = Factorised<ChessBuckets, Chess768>;
impl ChessBucketsFactorised {
    pub fn new(buckets: [usize; 64]) -> Self {
        Self::from_parts(ChessBuckets::new(buckets), Chess768)
    }
}

pub type ChessBucketsMirroredFactorised = Factorised<ChessBucketsMirrored, Chess768>;
impl ChessBucketsMirroredFactorised {
    pub fn new(buckets: [usize; 32]) -> Self {
        Self::from_parts(ChessBucketsMirrored::new(buckets), Chess768)
    }
}

pub type ChessBucketsMergedKingsFactorised = Factorised<ChessBucketsMergedKings, Chess768>;
impl ChessBucketsMergedKingsFactorised {
    pub fn new(buckets: [usize; 64]) -> Self {
        Self::from_parts(ChessBucketsMergedKings::new(buckets), Chess768)
    }
}

pub type ChessBucketsMergedKingsMirroredFactorised = Factorised<ChessBucketsMergedKingsMirrored, Chess768>;
impl ChessBucketsMergedKingsMirroredFactorised {
    pub fn new(buckets: [usize; 32]) -> Self {
        Self::from_parts(ChessBucketsMergedKingsMirrored::new(buckets), Chess768)
    }
}

pub trait SparseInputType: Clone + Send + Sync + 'static {
    type RequiredDataType: LoadableDataType + Send + Sync;

    /// The total number of inputs
    fn num_inputs(&self) -> usize;

    /// The maximum number of active inputs
    fn max_active(&self) -> usize;

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F);

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String;

    /// Description of the input type
    fn description(&self) -> String;

    fn is_factorised(&self) -> bool {
        false
    }

    fn merge_factoriser(&self, unmerged: Vec<f32>) -> Vec<f32> {
        assert!(self.is_factorised());
        unmerged
    }
}

fn get_num_buckets<const N: usize>(arr: &[usize; N]) -> usize {
    let mut max = 0;
    for &val in arr {
        max = max.max(val)
    }
    max + 1
}
