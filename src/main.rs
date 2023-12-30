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
        .ft(768)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    //trainer.load_from_checkpoint("checkpoints/net-gpu-epoch1");

    let mut schedule = TrainingSchedule {
        net_id: "net".to_string(),
        start_epoch: 1,
        num_epochs: 20,
        wdl_scheduler: WdlScheduler::new(0.2, 0.5),
        lr_scheduler: LrScheduler::new(0.001, 0.1, LrSchedulerType::Step(8)),
        save_rate: 1,
    };

    println!("Network Architecture: {trainer}");

    run_training(&mut trainer, &mut schedule, 4, "../../data/wha.data");
}