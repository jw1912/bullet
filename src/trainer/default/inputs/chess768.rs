use bulletformat::{chess::BoardIter, ChessBoard};

use super::InputType;

#[derive(Clone, Copy, Debug, Default)]
pub struct Chess768;
impl InputType for Chess768 {
    type RequiredDataType = ChessBoard;
    type FeatureIter = Chess768Iter;

    fn max_active_inputs(&self) -> usize {
        32
    }

    fn inputs(&self) -> usize {
        768
    }

    fn buckets(&self) -> usize {
        1
    }

    fn feature_iter(&self, pos: &Self::RequiredDataType) -> Self::FeatureIter {
        Chess768Iter { board_iter: pos.into_iter() }
    }
}

pub struct Chess768Iter {
    board_iter: BoardIter,
}

impl Iterator for Chess768Iter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.board_iter.next().map(|(piece, square)| {
            let c = usize::from(piece & 8 > 0);
            let pc = 64 * usize::from(piece & 7);
            let sq = usize::from(square);
            let wfeat = [0, 384][c] + pc + sq;
            let bfeat = [384, 0][c] + pc + (sq ^ 56);
            (wfeat, bfeat)
        })
    }
}
