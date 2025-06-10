mod adapter;
mod ataxx147;
mod chess768;
mod chess_buckets;
mod chess_buckets_mk;
mod factorised;

#[allow(deprecated)]
mod legacy;

pub use adapter::MarlinFormatAdapter;
pub use ataxx147::{Ataxx147, Ataxx98};
pub use chess768::Chess768;
pub use chess_buckets::{ChessBuckets, ChessBucketsMirrored};
pub use factorised::{Factorised, Factorises};

#[allow(deprecated)]
pub use chess_buckets_mk::*;

#[allow(deprecated)]
pub use legacy::InputType;

#[deprecated(note = "See `examples/progression/3_input_buckets.rs` for a faster alternative to this.")]
pub type ChessBucketsFactorised = Factorised<ChessBuckets, Chess768>;

#[allow(deprecated)]
impl ChessBucketsFactorised {
    pub fn new(buckets: [usize; 64]) -> Self {
        Self::from_parts(ChessBuckets::new(buckets), Chess768)
    }
}

#[deprecated(note = "See `examples/progression/3_input_buckets.rs` for a faster alternative to this.")]
pub type ChessBucketsMirroredFactorised = Factorised<ChessBucketsMirrored, Chess768>;

#[allow(deprecated)]
impl ChessBucketsMirroredFactorised {
    pub fn new(buckets: [usize; 32]) -> Self {
        Self::from_parts(ChessBucketsMirrored::new(buckets), Chess768)
    }
}

pub trait SparseInputType: Clone + Send + Sync + 'static {
    type RequiredDataType: Send + Sync;

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

pub const fn get_num_buckets<const N: usize>(arr: &[usize; N]) -> usize {
    let mut max = 0;
    let mut i = 0;

    while i < N {
        if arr[i] > max {
            max = arr[i];
        }

        i += 1;
    }
    max + 1
}
