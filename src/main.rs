use bullet::{data::Data, arch::NNUEParams, gd_tune, quantise::QuantisedNNUE};

pub const NET_NAME: &str = "maiden";

struct Rand(u32);
impl Rand {
    fn rand(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
	    self.0 ^= self.0 >> 17;
	    self.0 ^= self.0 << 5;
        f64::from(self.0) / f64::from(u32::MAX) - 1.
    }
}

fn main() -> std::io::Result<()> {
    let file_name = String::from("wha.epd");

    // initialise data
    let mut data = Data::default();
    data.1 = 4;
    data.add_contents(&file_name);

    // provide random starting parameters
    let mut params = Box::<NNUEParams>::default();
    let mut gen = Rand(173645501);
    for param in params.feature_weights.iter_mut() {
        *param = gen.rand();
    }
    for param in params.feature_bias.iter_mut() {
        *param = gen.rand();
    }
    for param in params.output_weights.iter_mut() {
        *param = gen.rand();
    }
    params.output_bias = gen.rand();

    // carry out tuning
    gd_tune(&data, &mut params, 2000, 0.01, NET_NAME);

    QuantisedNNUE::from_unquantised(&params).write_to_bin(&format!("{NET_NAME}-final.bin"))?;

    // exit
    Ok(())
}