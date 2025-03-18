use bulletformat::{chess::MarlinFormat, ChessBoard};

use super::SparseInputType;

#[derive(Clone)]
pub struct MarlinFormatAdapter<T: Clone>(pub T);

impl<T: SparseInputType<RequiredDataType = ChessBoard>> SparseInputType for MarlinFormatAdapter<T> {
    type RequiredDataType = MarlinFormat;

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        self.0.map_features(&ChessBoard::from(*pos), f);
    }

    fn max_active(&self) -> usize {
        self.0.max_active()
    }

    fn num_inputs(&self) -> usize {
        self.0.num_inputs()
    }

    fn description(&self) -> String {
        self.0.description()
    }

    fn shorthand(&self) -> String {
        self.0.shorthand()
    }

    fn is_factorised(&self) -> bool {
        self.0.is_factorised()
    }

    fn merge_factoriser(&self, unmerged: Vec<f32>) -> Vec<f32> {
        self.0.merge_factoriser(unmerged)
    }
}
