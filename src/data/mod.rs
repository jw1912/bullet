mod chess;
mod marlinformat;

pub use chess::ChessBoard;
pub use marlinformat::MarlinFormat;

trait DataType {
    type FeatureType;
}

#[derive(Default)]
pub struct Features {
    features: [(usize, usize); 32],
    len: usize,
    consumed: usize
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