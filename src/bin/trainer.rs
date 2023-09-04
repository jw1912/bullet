use bullet::{
    network::{NNUEParams, QuantisedNNUE, FEATURE_BIAS, OUTPUT_BIAS, OUTPUT_WEIGHTS},
    rng::Rand,
    trainer::{
        scheduler::{LrScheduler, SchedulerType},
        Trainer,
    },
    ActivationUsed, OptimiserUsed,
};

fn main() {
    let mut args = std::env::args();
    args.next();

    // all of these will be provided and validated by the `run.py` script
    let file_path = args.next().unwrap();
    let net_name = &args.next().unwrap();
    let threads = args.next().unwrap().parse().unwrap();
    let lr_start = args.next().unwrap().parse().unwrap();
    let blend = args.next().unwrap().parse().unwrap();
    let max_epochs: usize = args.next().unwrap().parse().unwrap();
    let batch_size = args.next().unwrap().parse().unwrap();
    let save_rate = args.next().unwrap().parse().unwrap();
    let skip_prop = args.next().unwrap().parse().unwrap();
    let lr_end: f32 = args.next().unwrap().parse().unwrap();
    let lr_step = args.next().unwrap().parse().unwrap();
    let lr_drop = args.next().unwrap().parse().unwrap();
    let lr_gamma = args.next().unwrap().parse().unwrap();
    let scale = args.next().unwrap().parse().unwrap();
    let cbcs = args.next().unwrap().parse().unwrap();

    let mut scheduler = LrScheduler::new(lr_start, 1.0, SchedulerType::Drop(1000));

    if lr_end != 0.0 {
        scheduler.set_type(SchedulerType::Step(1));
        let gamma = (lr_start / lr_end).ln() / (max_epochs - 1).max(1) as f32;
        scheduler.set_gamma((-gamma).exp());
    }

    if lr_step != 0 {
        scheduler.set_type(SchedulerType::Step(lr_step));
        scheduler.set_gamma(lr_gamma);
    }

    if lr_drop != 0 {
        scheduler.set_type(SchedulerType::Drop(lr_drop));
        scheduler.set_gamma(lr_gamma);
    }

    let optimiser = OptimiserUsed::default();

    let mut trainer = Trainer::new(file_path, threads, scheduler, blend, skip_prop, optimiser);

    // provide random starting parameters
    let mut params = NNUEParams::new();
    let mut gen = Rand::new(173645501);
    for param in params[..FEATURE_BIAS].iter_mut() {
        *param = gen.rand(0.01);
    }

    for param in params[OUTPUT_WEIGHTS..OUTPUT_BIAS].iter_mut() {
        *param = gen.rand(0.01);
    }

    // carry out tuning
    trainer.run::<ActivationUsed>(
        &mut params,
        max_epochs,
        net_name,
        save_rate,
        batch_size,
        scale,
        cbcs,
    );

    // safe to bin file
    QuantisedNNUE::from_unquantised(&params)
        .write_to_bin(&format!("nets/{net_name}.bin"))
        .expect("Should never fail!");

    println!("Saved [nets/{net_name}.bin]");
}
