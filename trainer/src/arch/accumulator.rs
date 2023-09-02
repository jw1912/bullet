use super::nnue::{HIDDEN, NNUE, FEATURE_BIAS, OUTPUT_WEIGHTS};

#[derive(Clone, Copy)]
pub struct Accumulator<T> {
    vals: [T; HIDDEN],
}

impl<T> std::ops::Deref for Accumulator<T> {
    type Target = [T; HIDDEN];
    fn deref(&self) -> &Self::Target {
        &self.vals
    }
}

impl<T> std::ops::Index<usize> for Accumulator<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.vals[index]
    }
}

impl<T> std::ops::IndexMut<usize> for Accumulator<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.vals[index]
    }
}

impl<T: Copy + Default> Accumulator<T> {
    pub fn new(vals: [T; HIDDEN]) -> Self {
        Self { vals }
    }

    pub fn load_biases(nnue: &NNUE<T>) -> Self {
        let bias = &nnue[FEATURE_BIAS..OUTPUT_WEIGHTS];
        let mut hmm = [T::default(); HIDDEN];
        hmm.copy_from_slice(bias);
        Self::new(hmm)
    }
}

impl<T: Copy + std::ops::AddAssign<T>> Accumulator<T> {
    pub fn add_feature(&mut self, feature_idx: usize, nnue: &NNUE<T>) {
        let start = feature_idx * HIDDEN;
        for (i, d) in self.vals.iter_mut().zip(&nnue[start..start + HIDDEN]) {
            *i += *d;
        }
    }
}
