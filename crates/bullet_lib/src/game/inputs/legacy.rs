use crate::value::loader::LoadableDataType;

use super::SparseInputType;

#[deprecated]
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

impl<T: InputType> SparseInputType for T {
    type RequiredDataType = <Self as InputType>::RequiredDataType;

    fn num_inputs(&self) -> usize {
        self.size()
    }

    fn max_active(&self) -> usize {
        self.max_active_inputs()
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        for (stm, ntm) in self.feature_iter(pos) {
            f(stm, ntm);
        }
    }

    fn shorthand(&self) -> String {
        format!("{}x{}", self.inputs(), self.buckets())
    }

    fn description(&self) -> String {
        <Self as InputType>::description(self)
    }

    fn is_factorised(&self) -> bool {
        <Self as InputType>::is_factorised(self)
    }

    fn merge_factoriser(&self, unmerged: Vec<f32>) -> Vec<f32> {
        <Self as InputType>::merge_factoriser(self, unmerged)
    }
}
