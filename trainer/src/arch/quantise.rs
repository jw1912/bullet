use super::nnue::{NNUEParams, NNUE, OUTPUT_BIAS, OUTPUT_WEIGHTS};

const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

pub type QuantisedNNUE = NNUE<i16>;

impl QuantisedNNUE {
    pub fn from_unquantised(nnue: &NNUEParams) -> Box<Self> {
        let mut res = QuantisedNNUE::new();

        for (q, &param) in res
            .iter_mut()
            .take(OUTPUT_WEIGHTS)
            .zip(nnue.iter().take(OUTPUT_WEIGHTS))
        {
            *q = (param * (QA as f32)) as i16;
        }

        for (q, &param) in res
            .iter_mut()
            .take(OUTPUT_BIAS)
            .skip(OUTPUT_WEIGHTS)
            .zip(nnue.iter().take(OUTPUT_BIAS).skip(OUTPUT_WEIGHTS))
        {
            *q = (param * (QB as f32)) as i16;
        }

        res[OUTPUT_BIAS] = (nnue[OUTPUT_BIAS] * (QAB as f32)) as i16;

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
