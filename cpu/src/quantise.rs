use super::{NetworkParams, Network, NETWORK_SIZE, OUTPUT_BIAS, OUTPUT_WEIGHTS};
use common::util::write_to_bin;

pub fn quantise_and_write(nnue: &NetworkParams, net_path: &str, qa: i32, qb: i32) {
    let qnnue = QuantisedNetwork::from_unquantised(nnue, qa, qb);
    qnnue.write_to_bin(net_path).unwrap();
}

type QuantisedNetwork = Network<i16>;

impl QuantisedNetwork {
    fn from_unquantised(nnue: &NetworkParams, qa: i32, qb: i32) -> Box<Self> {
        let mut res = QuantisedNetwork::new();

        for i in 0..OUTPUT_WEIGHTS {
            res[i] = (nnue[i] * (qa as f32)) as i16;
        }

        for i in OUTPUT_WEIGHTS..OUTPUT_BIAS {
            res[i] = (nnue[i] * (qb as f32)) as i16;
        }

        let qab = qa * qb;

        for i in OUTPUT_BIAS..NETWORK_SIZE {
            res[i] = (nnue[i] * (qab as f32)) as i16;
        }

        res
    }

    fn write_to_bin(&self, output_path: &str) -> std::io::Result<()> {
        const SIZEOF: usize = std::mem::size_of::<QuantisedNetwork>();
        write_to_bin::<Self, SIZEOF>(self, output_path, true)
    }
}
