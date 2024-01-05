use bullet::{
    inputs, Activation, LocalSettings, LrScheduler, TrainerBuilder, TrainingSchedule, WdlScheduler,
};

fn main() {
    let mut trainer = TrainerBuilder::default()
        .set_batch_size(16_384)
        .set_eval_scale(400.0)
        .set_input(inputs::Chess768)
        .ft(768)
        .activate(Activation::CReLU)
        .add_layer(8)
        .activate(Activation::ReLU)
        .add_layer(16)
        .activate(Activation::ReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "net-05.01.24".to_string(),
        start_epoch: 1,
        end_epoch: 17,
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

    let settings = LocalSettings {
        threads: 4,
        data_file_path: "../../data/akimbo3-9.data",
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);
}
