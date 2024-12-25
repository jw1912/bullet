mod ataxx147;
mod chess768;
mod chess_buckets;
mod chess_buckets_hm;
mod factorised;

pub use ataxx147::{Ataxx147, Ataxx98};
pub use chess768::Chess768;
pub use chess_buckets::ChessBuckets;
pub use chess_buckets_hm::{ChessBucketsMirrored, ChessBucketsMirroredFactorised};
pub use factorised::{Factorised, Factorises};

use super::loader::LoadableDataType;

pub trait InputType: Send + Sync + Copy + Default + 'static {
    type RequiredDataType: LoadableDataType + Copy + Send + Sync;
    type FeatureIter: Iterator<Item = (usize, usize)>;

    fn max_active_inputs(&self) -> usize;

    /// The number of inputs per bucket.
    fn inputs(&self) -> usize;

    /// The number of buckets.
    /// ### Note
    /// This is purely aesthetic, training is completely unchanged
    /// so long as `inputs * buckets` is constant.
    fn buckets(&self) -> usize;

    fn size(&self) -> usize {
        self.inputs() * self.buckets()
    }

    fn feature_iter(&self, pos: &Self::RequiredDataType) -> Self::FeatureIter;

    fn is_factorised(&self) -> bool {
        false
    }

    fn merge_factoriser(&self, unmerged: Vec<f32>) -> Vec<f32> {
        assert!(self.is_factorised());
        unmerged
    }

    fn description(&self) -> String {
        "Unspecified input format".to_string()
    }
}

fn get_num_buckets<const N: usize>(arr: &[usize; N]) -> usize {
    let mut max = 0;
    for &val in arr {
        max = max.max(val)
    }
    max + 1
}
