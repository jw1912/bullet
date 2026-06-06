mod adapter;
mod ataxx147;
mod chess768;
mod chess_buckets;
mod factorised;

pub use adapter::MarlinFormatAdapter;
pub use ataxx147::{Ataxx98, Ataxx147};
pub use chess_buckets::{ChessBuckets, ChessBucketsMirrored};
pub use chess768::Chess768;
pub use factorised::{Factorised, Factorises};

pub trait SparseInputType: Clone + Send + Sync + 'static {
    type RequiredDataType: Copy + Send + Sync;

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
