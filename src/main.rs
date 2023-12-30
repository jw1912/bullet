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
    let mut net = TrainerBuilder::<inputs::Chess768>::default()
        .set_batch_size(16_384)
        .set_eval_scale(400.0)
        .ft(512)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let mut schedule = TrainingSchedule::new(
        20,
        WdlScheduler::new(0.2, 0.5),
        LrScheduler::new(0.001, 0.1, LrSchedulerType::Step(8)),
        1,
    );

    println!("Network Architecture: {net}");

    run_training(&mut net, &mut schedule, 4, "../../data/wha.data", false, 1);
}