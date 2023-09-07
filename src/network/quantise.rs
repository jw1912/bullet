use crate::{Data, HIDDEN, Input, network::{InputType, FEATURE_BIAS}, data::DataType};
use super::{NNUEParams, NNUE, OUTPUT_BIAS, OUTPUT_WEIGHTS, NNUE_SIZE};

const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

pub fn quantise_and_write(nnue: &NNUEParams, net_path: &str) {
    if Input::FACTORISER {
        let qfnnue = QuantisedFactorisedNNUE::from_unquantised(nnue);
        qfnnue.write_to_bin(net_path).unwrap();
    } else {
        let qnnue = QuantisedNNUE::from_unquantised(nnue);
        qnnue.write_to_bin(net_path).unwrap();
    }
}

type QuantisedNNUE = NNUE<i16>;

impl QuantisedNNUE {
    fn from_unquantised(nnue: &NNUEParams) -> Box<Self> {
        let mut res = QuantisedNNUE::new();

        for i in 0..OUTPUT_WEIGHTS {
            res[i] = (nnue[i] * (QA as f32)) as i16;
        }

        for i in OUTPUT_WEIGHTS..OUTPUT_BIAS {
            res[i] = (nnue[i] * (QB as f32)) as i16;
        }

        res[OUTPUT_BIAS] = (nnue[OUTPUT_BIAS] * (QAB as f32)) as i16;

        res
    }

    fn write_to_bin(&self, output_path: &str) -> std::io::Result<()> {
        use std::io::Write;
        const SIZEOF: usize = std::mem::size_of::<QuantisedNNUE>();

        let mut file = std::fs::File::create(output_path)?;

        unsafe {
            let ptr: *const Self = self;
            let slice_ptr: *const u8 = std::mem::transmute(ptr);
            let slice = std::slice::from_raw_parts(slice_ptr, SIZEOF);
            file.write_all(slice)?;
        }
        Ok(())
    }
}

pub struct QuantisedFactorisedNNUE {
    weights: [i16; NNUE_SIZE - Data::INPUTS * HIDDEN],
}

impl QuantisedFactorisedNNUE {
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

    fn from_unquantised(nnue: &NNUEParams) -> Box<Self> {
        let mut res = Self::new();

        const OFFSET: usize = Data::INPUTS * HIDDEN * Input::FACTORISER as usize;
        const NEW_HIDDEN: usize = FEATURE_BIAS - OFFSET;

        for i in 0..NEW_HIDDEN {
            res.weights[i] = ((nnue[i % Data::INPUTS] + nnue[OFFSET + i]) * (QA as f32)) as i16;
        }

        for i in FEATURE_BIAS..OUTPUT_WEIGHTS {
            res.weights[i - OFFSET] = (nnue[i] * (QA as f32)) as i16;
        }

        for i in OUTPUT_WEIGHTS..OUTPUT_BIAS {
            res.weights[i - OFFSET] = (nnue[i] * (QB as f32)) as i16;
        }

        res.weights[OUTPUT_BIAS - OFFSET] = (nnue[OUTPUT_BIAS] * (QAB as f32)) as i16;

        res
    }

    fn write_to_bin(&self, output_path: &str) -> std::io::Result<()> {
        use std::io::Write;
        const SIZEOF: usize = std::mem::size_of::<QuantisedFactorisedNNUE>();

        let mut file = std::fs::File::create(output_path)?;

        unsafe {
            let ptr: *const Self = self;
            let slice_ptr: *const u8 = std::mem::transmute(ptr);
            let slice = std::slice::from_raw_parts(slice_ptr, SIZEOF);
            file.write_all(slice)?;
        }
        Ok(())
    }
}
