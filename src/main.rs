use bullet::{
    arch::{NNUEParams, QuantisedNNUE},
    trainer::Trainer,
};

pub const NET_NAME: &str = "net1";

struct Rand(u32);
impl Rand {
    fn rand(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (1. - f64::from(self.0) / f64::from(u32::MAX)) / 100.
    }
}

fn main() {
    let file_name = std::env::args().nth(1).expect("Expected a file name!");

    // initialise data
    let mut trainer = Trainer::new(6);
    trainer.add_data(&file_name);

    // provide random starting parameters
    let mut params = Box::<NNUEParams>::default();
    let mut gen = Rand(173645501);
    for param in params.feature_weights.iter_mut() {
        *param = gen.rand();
    }

    for param in params.output_weights.iter_mut() {
        *param = gen.rand();
    }

    // carry out tuning
    trainer.run(&mut params, 1000, 0.001, NET_NAME, 1, 10);

    // safe to bin file
    QuantisedNNUE::from_unquantised(&params)
        .write_to_bin(&format!("{NET_NAME}.bin"))
        .expect("Should never fail!");
}
