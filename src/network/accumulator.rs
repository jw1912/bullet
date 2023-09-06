use super::{FEATURE_BIAS, HIDDEN, OUTPUT_WEIGHTS, NNUEParams};

#[derive(Clone, Copy)]
pub struct Accumulator {
    vals: [f32; HIDDEN],
}

impl std::ops::Deref for Accumulator {
    type Target = [f32; HIDDEN];
    fn deref(&self) -> &Self::Target {
        &self.vals
    }
}

impl std::ops::Index<usize> for Accumulator {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.vals[index]
    }
}

impl std::ops::IndexMut<usize> for Accumulator {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.vals[index]
    }
}

impl Accumulator {
    pub fn new(vals: [f32; HIDDEN]) -> Self {
        Self { vals }
    }

    pub fn load_biases(nnue: &NNUEParams) -> Self {
        let bias = &nnue[FEATURE_BIAS..OUTPUT_WEIGHTS];
        let mut hmm = [0.0; HIDDEN];
        hmm.copy_from_slice(bias);
        Self::new(hmm)
    }
}

impl Accumulator {
    pub fn add_feature(&mut self, feature_idx: usize, nnue: &NNUEParams) {
        let start = feature_idx * HIDDEN;
        for (i, d) in self.vals.iter_mut().zip(&nnue[start..start + HIDDEN]) {
            *i += *d;
        }
    }
}
