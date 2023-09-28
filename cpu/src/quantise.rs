use super::{NetworkParams, Network, NETWORK_SIZE, OUTPUT_BIAS, OUTPUT_WEIGHTS, FEATURE_BIAS};
use common::{
    data::DataType,
    inputs::InputType,
    Data, Input, HIDDEN,
    util::write_to_bin,
};

const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

pub fn quantise_and_write(nnue: &NetworkParams, net_path: &str) {
    if Input::FACTORISER {
        let qfnnue = QuantisedFactorisedNetwork::from_unquantised(nnue);
        qfnnue.write_to_bin(net_path).unwrap();
    } else {
        let qnnue = QuantisedNetwork::from_unquantised(nnue);
        qnnue.write_to_bin(net_path).unwrap();
    }
}

type QuantisedNetwork = Network<i16>;

impl QuantisedNetwork {
    fn from_unquantised(nnue: &NetworkParams) -> Box<Self> {
        let mut res = QuantisedNetwork::new();

        for i in 0..OUTPUT_WEIGHTS {
            res[i] = (nnue[i] * (QA as f32)) as i16;
        }

        for i in OUTPUT_WEIGHTS..OUTPUT_BIAS {
            res[i] = (nnue[i] * (QB as f32)) as i16;
        }

        for i in OUTPUT_BIAS..NETWORK_SIZE {
            res[i] = (nnue[i] * (QAB as f32)) as i16;
        }

        res
    }

    fn write_to_bin(&self, output_path: &str) -> std::io::Result<()> {
        const SIZEOF: usize = std::mem::size_of::<QuantisedNetwork>();
        write_to_bin::<Self, SIZEOF>(self, output_path, true)
    }
}

pub struct QuantisedFactorisedNetwork {
    weights: [i16; NETWORK_SIZE - Data::INPUTS * HIDDEN * Input::FACTORISER as usize],
}

impl QuantisedFactorisedNetwork {
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

    fn from_unquantised(nnue: &NetworkParams) -> Box<Self> {
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

        for i in OUTPUT_BIAS..NETWORK_SIZE {
            res.weights[i - OFFSET] = (nnue[i] * (QAB as f32)) as i16;
        }

        res
    }

    fn write_to_bin(&self, output_path: &str) -> std::io::Result<()> {
        const SIZEOF: usize = std::mem::size_of::<QuantisedFactorisedNetwork>();
        write_to_bin::<Self, SIZEOF>(self, output_path, true)
    }
}
