use crate::{
    Data,
    Input,
    data::MAX_FEATURES,
    inputs::InputType, OutputBucket,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoardCUDA {
    features: [u16; MAX_FEATURES],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CudaResult {
    pub score: f32,
    pub bucket: u32,
}

impl Default for BoardCUDA {
    fn default() -> Self {
        Self { features: [0; MAX_FEATURES] }
    }
}

impl BoardCUDA {
    pub fn len() -> usize {
        MAX_FEATURES
    }

    pub fn push(
        board: &Data,
        our_inputs: &mut Vec<BoardCUDA>,
        opp_inputs: &mut Vec<BoardCUDA>,
        results: &mut Vec<CudaResult>,
        blend: f32,
        scale: f32,
    ) {
        let mut our_board = BoardCUDA::default();
        let mut opp_board = BoardCUDA::default();

        let mut i = 0;
        let mut idx = 0;

        for feat in board.into_iter() {
            let (wfeat, bfeat) = Input::get_feature_indices(feat);

            OutputBucket::update_output_bucket(&mut idx, usize::from(feat.0 & 7));

            our_board.features[i] = wfeat as u16;
            opp_board.features[i] = bfeat as u16;
            i += 1;
        }

        let bucket = OutputBucket::get_bucket(idx);

        if i < MAX_FEATURES {
            our_board.features[i] = u16::MAX;
            opp_board.features[i] = u16::MAX;
        }

        let result = CudaResult {
            score: board.blended_result(blend, scale),
            bucket: bucket as u32,
        };

        our_inputs.push(our_board);
        opp_inputs.push(opp_board);
        results.push(result);
    }
}