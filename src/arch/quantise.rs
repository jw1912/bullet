use super::nnue::{NNUE, NNUEParams};

const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

pub type QuantisedNNUE = NNUE<i16>;

impl QuantisedNNUE {
    pub fn from_unquantised(nnue: &NNUEParams) -> Box<Self> {
        let mut res = Box::<QuantisedNNUE>::default();

        for (i, &param) in nnue.feature_weights.iter().enumerate() {
            res.feature_weights[i] = (param * f64::from(QA)) as i16;
        }

        for (i, &param) in nnue.feature_bias.iter().enumerate() {
            res.feature_bias[i] = (param * f64::from(QA)) as i16;
        }

        for (i, &param) in nnue.output_weights.iter().enumerate() {
            res.output_weights[i] = (param * f64::from(QB)) as i16;
        }

        res.output_bias = (nnue.output_bias * f64::from(QAB)) as i16;

        res
    }

    pub fn write_to_bin(&self, output_path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(output_path)?;
        const SIZEOF: usize = std::mem::size_of::<QuantisedNNUE>();
        unsafe {
            file.write_all(
                &std::mem::transmute::<QuantisedNNUE, [u8; SIZEOF]>(self.clone())
            )?;
        }
        Ok(())
    }
}
