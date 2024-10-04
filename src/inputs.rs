use bulletformat::BulletFormat;

mod ataxx147;
mod chess768;
mod chess_buckets;
mod chess_buckets_hm;

pub use ataxx147::{Ataxx147, Ataxx98};
pub use chess768::Chess768;
pub use chess_buckets::ChessBuckets;
pub use chess_buckets_hm::{ChessBucketsMirrored, ChessBucketsMirroredFactorised};

pub trait InputType: Send + Sync + Copy + Default + 'static {
    type RequiredDataType: BulletFormat + Copy + Send + Sync;
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
}

fn get_num_buckets<const N: usize>(arr: &[usize; N]) -> usize {
    let mut max = 0;
    for &val in arr {
        max = max.max(val)
    }
    max + 1
}
