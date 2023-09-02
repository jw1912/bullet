use std::ops::{AddAssign, Deref, DerefMut};

pub const HIDDEN: usize = crate::HIDDEN_SIZE;
pub const INPUT: usize = 768;
pub const K: f32 = 1.0;
pub const K4: f32 = K / 400.0;

pub type NNUEParams = NNUE<f32>;

const NNUE_SIZE: usize = (INPUT + 3) * HIDDEN + 1;
pub const FEATURE_BIAS: usize = INPUT * HIDDEN;
pub const OUTPUT_WEIGHTS: usize = (INPUT + 1) * HIDDEN;
pub const OUTPUT_BIAS: usize = (INPUT + 3) * HIDDEN;

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
#[repr(C)]
pub struct NNUE<T> {
    pub weights: [T; NNUE_SIZE],
}

impl<T> NNUE<T> {
    pub fn new() -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }
}

impl<T: AddAssign<T> + Copy> AddAssign<&NNUE<T>> for NNUE<T> {
    fn add_assign(&mut self, rhs: &NNUE<T>) {
        for (i, &j) in self.weights.iter_mut().zip(rhs.weights.iter()) {
            *i += j;
        }
    }
}

impl<T> Deref for NNUE<T> {
    type Target = [T; NNUE_SIZE];
    fn deref(&self) -> &Self::Target {
        &self.weights
    }
}

impl<T> DerefMut for NNUE<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.weights
    }
}
