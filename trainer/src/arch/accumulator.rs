use super::nnue::NNUE;

#[derive(Clone, Copy)]
pub struct Accumulator<T, const SIZE: usize> {
    vals: [T; SIZE],
}

impl<T, const SIZE: usize> std::ops::Deref for Accumulator<T, SIZE> {
    type Target = [T; SIZE];
    fn deref(&self) -> &Self::Target {
        &self.vals
    }
}

impl<T, const SIZE: usize> std::ops::Index<usize> for Accumulator<T, SIZE> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.vals[index]
    }
}

impl<T, const SIZE: usize> std::ops::IndexMut<usize> for Accumulator<T, SIZE> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.vals[index]
    }
}

impl<T, const SIZE: usize> Accumulator<T, SIZE> {
    pub fn new(vals: [T; SIZE]) -> Self {
        Self { vals }
    }
}

impl<T: Copy + std::ops::AddAssign<T>, const SIZE: usize> Accumulator<T, SIZE> {
    pub fn add_feature(&mut self, feature_idx: usize, nnue: &NNUE<T>) {
        let start = feature_idx * SIZE;
        for (i, d) in self.vals.iter_mut().zip(&nnue[start..start + SIZE]) {
            *i += *d;
        }
    }
}
