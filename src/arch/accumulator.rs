use std::ops::{AddAssign, Index, IndexMut};
use super::nnue::NNUE;

#[derive(Clone, Copy)]
pub struct Accumulator<T, const SIZE: usize>(pub [T; SIZE]);

impl<T, const SIZE: usize> Index<usize> for Accumulator<T, SIZE> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, const SIZE: usize> IndexMut<usize> for Accumulator<T, SIZE> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T: Copy + AddAssign<T>, const SIZE: usize> Accumulator<T, SIZE> {
    pub fn new(vals: [T; SIZE]) -> Self {
        Self(vals)
    }

    pub fn add_feature(&mut self, feature_idx: usize, nnue: &NNUE<T>) {
        let start = feature_idx * SIZE;
        for (i, d) in self.0.iter_mut().zip(&nnue.feature_weights[start..start + SIZE]) {
            *i += *d;
        }
    }
}