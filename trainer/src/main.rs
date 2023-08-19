use trainer::{
    ActivationUsed,
    OptimiserUsed,
    arch::{NNUEParams, QuantisedNNUE},
    rng::Rand,
    trainer::Trainer,
};

fn main() {
    let mut args = std::env::args();
    println!("Beginning Training");
    args.next();

    // all of these will be provided and validated by the `run.py` script
    let file_path = args.next().unwrap();
    let threads = args.next().unwrap().parse().unwrap();
    let lr = args.next().unwrap().parse().unwrap();
    let blend = args.next().unwrap().parse().unwrap();
    let max_epochs = args.next().unwrap().parse().unwrap();
    let batch_size = args.next().unwrap().parse().unwrap();
    let save_rate = args.next().unwrap().parse().unwrap();
    let net_name = &args.next().unwrap();

    let optimiser = OptimiserUsed::default();

    let mut trainer = Trainer::new(file_path, threads, lr, blend, optimiser);

    // provide random starting parameters
    let mut params = NNUEParams::new();
    let mut gen = Rand::new(173645501);
    for param in params.feature_weights.iter_mut() {
        *param = gen.rand(0.01);
    }

    for param in params.output_weights.iter_mut() {
        *param = gen.rand(0.01);
    }

    // carry out tuning
    trainer.run::<ActivationUsed>(
        &mut params,
        max_epochs,
        net_name,
        save_rate,
        batch_size,
    );

    // safe to bin file
    QuantisedNNUE::from_unquantised(&params)
        .write_to_bin(&format!("{net_name}.bin"))
        .expect("Should never fail!");
}
