use bullet::{
    inputs, Activation, LrScheduler, TrainerBuilder, TrainingSchedule, WdlScheduler,
};

fn main() {
    let mut trainer = TrainerBuilder::default()
        .set_batch_size(16_384)
        .set_eval_scale(400.0)
        .set_quantisations(&[181, 128, 64])
        .set_input(inputs::Chess768)
        .ft(768)
        .activate(Activation::SCReLU)
        .add_layer(16)
        .activate(Activation::ReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "morelayers".to_string(),
        start_epoch: 1,
        end_epoch: 20,
        wdl_scheduler: WdlScheduler::Linear {
            start: 0.2,
            end: 0.5,
        },
        lr_scheduler: LrScheduler::Step {
            start: 0.001,
            gamma: 0.1,
            step: 8,
        },
        save_rate: 1,
    };

    trainer.run(
        &schedule,
        4,
        "../../data/akimbo3-9.data",
        "checkpoints",
    );
}
