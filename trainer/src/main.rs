use trainer::{
    ActivationUsed,
    arch::{NNUEParams, QuantisedNNUE},
    trainer::Trainer,
};

// Saving Net
const NET_NAME: &str = "net";
const REPORT_RATE: usize = 1;
const SAVE_RATE: usize = 101;

const THREADS: usize = 6;

// Training Hyperparams
const BLEND: f64 = 0.5;
const LR: f64 = 0.001;
const MAX_EPOCHS: usize = 65;
const BATCH_SIZE: usize = 16384;

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
    let file_path = std::env::args().nth(1).expect("Expected a file name!");

    // initialise data
    let mut trainer = Trainer::new(file_path, THREADS, LR, BLEND);

    // provide random starting parameters
    let mut params = NNUEParams::new();
    let mut gen = Rand(173645501);
    for param in params.feature_weights.iter_mut() {
        *param = gen.rand();
    }

    for param in params.output_weights.iter_mut() {
        *param = gen.rand();
    }

    // carry out tuning
    trainer.run::<ActivationUsed>(
        &mut params,
        MAX_EPOCHS,
        NET_NAME,
        REPORT_RATE,
        SAVE_RATE,
        BATCH_SIZE,
    );

    // safe to bin file
    QuantisedNNUE::from_unquantised(&params)
        .write_to_bin(&format!("{NET_NAME}.bin"))
        .expect("Should never fail!");
}
