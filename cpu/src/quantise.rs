use super::{NetworkParams, Network, NETWORK_SIZE, OUTPUT_BIAS, OUTPUT_WEIGHTS};
use common::util::write_to_bin;

const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

pub fn quantise_and_write(nnue: &NetworkParams, net_path: &str) {
    let qnnue = QuantisedNetwork::from_unquantised(nnue);
    qnnue.write_to_bin(net_path).unwrap();
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
