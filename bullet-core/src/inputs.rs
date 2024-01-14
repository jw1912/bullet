use bulletformat::BulletFormat;

mod ataxx147;
mod chess_buckets_hm;
mod chess_buckets;
mod chess768;

pub use ataxx147::Ataxx147;
pub use chess_buckets_hm::ChessBucketsMirrored;
pub use chess_buckets::ChessBuckets;
pub use chess768::Chess768;

pub trait InputType: Send + Sync + Copy + Default {
    type RequiredDataType: BulletFormat + Copy + Send + Sync;
    type FeatureIter: Iterator<Item = (usize, usize)>;

    fn max_active_inputs(&self) -> usize;

    fn inputs(&self) -> usize;

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
