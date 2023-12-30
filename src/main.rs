use bullet::{
    Activation,
    inputs,
    TrainerBuilder,
    run_training,
    TrainingSchedule,
    WdlScheduler,
    LrScheduler,
    LrSchedulerType,
};

fn main() {
    let mut trainer = TrainerBuilder::<inputs::Chess768>::default()
        .set_batch_size(16_384)
        .set_eval_scale(400.0)
        .ft(64)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    trainer.load_from_checkpoint("checkpoints/net-gpu-epoch1");

    let mut schedule = TrainingSchedule::new(
        "net-gpu-wha".to_string(),
        20,
        WdlScheduler::new(0.2, 0.5),
        LrScheduler::new(0.001, 0.1, LrSchedulerType::Step(8)),
        1,
    );

    println!("Network Architecture: {trainer}");

    run_training(
        &mut trainer,
        &mut schedule,
        4,
        "../../data/wha.data",
        false,
        2,
    );
}