use super::nnue::{NNUEParams, NNUE};

const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

pub type QuantisedNNUE = NNUE<i16>;

impl QuantisedNNUE {
    pub fn from_unquantised(nnue: &NNUEParams) -> Box<Self> {
        let mut res = QuantisedNNUE::new();

        for (i, &param) in nnue.feature_weights.iter().enumerate() {
            res.feature_weights[i] = (param * (QA as f32)) as i16;
        }

        for (i, &param) in nnue.feature_bias.iter().enumerate() {
            res.feature_bias[i] = (param * (QA as f32)) as i16;
        }

        for (i, &param) in nnue.output_weights.iter().enumerate() {
            res.output_weights[i] = (param * (QB as f32)) as i16;
        }

        res.output_bias = (nnue.output_bias * (QAB as f32)) as i16;

        res
    }

    pub fn write_to_bin(&self, output_path: &str) -> std::io::Result<()> {
        use std::{io::Write, mem::size_of};
        const SIZEOF: usize = size_of::<QuantisedNNUE>();

        let mut file = std::fs::File::create(output_path)?;

        unsafe {
            let ptr: *const QuantisedNNUE = self;
            let slice_ptr: *const u8 = std::mem::transmute(ptr);
            let slice = std::slice::from_raw_parts(slice_ptr, SIZEOF);
            file.write_all(slice)?;
        }
        Ok(())
    }
}
