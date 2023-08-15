use std::ops::AddAssign;

pub const HIDDEN: usize = crate::HIDDEN_SIZE;
pub const INPUT: usize = 768;
pub const K: f64 = 3.6;

pub type NNUEParams = NNUE<f64>;

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
#[repr(C)]
pub struct NNUE<T> {
    pub feature_weights: [T; INPUT * HIDDEN],
    pub feature_bias: [T; HIDDEN],
    pub output_weights: [T; 2 * HIDDEN],
    pub output_bias: T,
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

impl<T: AddAssign<T> + Copy> AddAssign<NNUE<T>> for NNUE<T> {
    fn add_assign(&mut self, rhs: NNUE<T>) {
        for (i, &j) in self
            .feature_weights
            .iter_mut()
            .zip(rhs.feature_weights.iter())
        {
            *i += j;
        }

        for (i, &j) in self
            .output_weights
            .iter_mut()
            .zip(rhs.output_weights.iter())
        {
            *i += j;
        }

        for (i, &j) in self.feature_bias.iter_mut().zip(rhs.feature_bias.iter()) {
            *i += j;
        }

        self.output_bias += rhs.output_bias;
    }
}
