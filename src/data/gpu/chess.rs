use crate::{
    data::{cpu::chess::ChessBoard, InputType, MAX_FEATURES},
    Input,
};


#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChessBoardCUDA {
    features: [u16; MAX_FEATURES],
}

impl Default for ChessBoardCUDA {
    fn default() -> Self {
        Self { features: [0; MAX_FEATURES] }
    }
}

impl ChessBoardCUDA {
    pub fn push(
        board: &ChessBoard,
        inputs: &mut Vec<ChessBoardCUDA>,
        results: &mut Vec<f32>,
        blend: f32,
        scale: f32,
    ) {
        let mut cuda_board = ChessBoardCUDA::default();

        let mut i = 0;

        for feat in board.into_iter() {
            let (wfeat, bfeat) = Input::get_feature_indices(feat);

            cuda_board.features[i] = wfeat as u16;
            cuda_board.features[i + 1] = bfeat as u16;
            i += 2;
            if Input::FACTORISER {
                cuda_board.features[i] = (wfeat % 768) as u16;
                cuda_board.features[i + 1] = (bfeat % 768) as u16;
                i += 2;
            }
        }

        if i < MAX_FEATURES {
            cuda_board.features[i] = u16::MAX;
        }

        inputs.push(cuda_board);
        results.push(board.blended_result(blend, scale));
    }
}