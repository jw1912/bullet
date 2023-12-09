pub mod gpu;

pub use bulletformat::{AtaxxBoard, BulletFormat, ChessBoard};
pub use gpu::{BoardCUDA, CudaResult};

use crate::Data;

pub const MAX_FEATURES: usize = Data::MAX_FEATURES;

#[derive(Debug)]
pub struct Features {
    features: [(usize, usize); MAX_FEATURES],
    len: usize,
    consumed: usize,
}

impl Default for Features {
    fn default() -> Self {
        Self {
            features: [(0, 0); MAX_FEATURES],
            len: 0,
            consumed: 0,
        }
    }
}

impl Features {
    pub fn push(&mut self, wfeat: usize, bfeat: usize) {
        self.features[self.len] = (wfeat, bfeat);
        self.len += 1;
    }
}

impl Iterator for Features {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.consumed == self.len {
            return None;
        }

        let ret = self.features[self.consumed];

        self.consumed += 1;

        Some(ret)
    }
}
