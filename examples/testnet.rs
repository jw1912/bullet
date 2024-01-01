use bullet::{
    inputs, run_training, Activation, LrScheduler, LrSchedulerType, TrainerBuilder,
    TrainingSchedule, WdlScheduler,
};

fn main() {
    let mut trainer = TrainerBuilder::default()
        .set_batch_size(16_384)
        .set_eval_scale(400.0)
        .set_quantisations(&[181, 64])
        .set_input(inputs::Chess768)
        .ft(32)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    trainer.load_from_checkpoint("checkpoints/testnet");

    let mut schedule = TrainingSchedule {
        net_id: "testnet".to_string(),
        start_epoch: 1,
        num_epochs: 5,
        wdl_scheduler: WdlScheduler::new(0.2, 0.2),
        lr_scheduler: LrScheduler::new(0.001, 0.1, LrSchedulerType::Step(8)),
        save_rate: 10,
    };

    run_training(
        &mut trainer,
        &mut schedule,
        4,
        "../../data/batch.data",
        "checkpoints",
    );
}
