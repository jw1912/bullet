/*
This is used to confirm non-functional changes for bullet.
*/
use bullet::{
    inputs, Activation, LocalSettings, LrScheduler, TrainerBuilder, TrainingSchedule, WdlScheduler,
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

    let schedule = TrainingSchedule {
        net_id: "testnet".to_string(),
        start_epoch: 1,
        end_epoch: 5,
        wdl_scheduler: WdlScheduler::Constant { value: 0.2 },
        lr_scheduler: LrScheduler::Constant { value: 0.001 },
        save_rate: 10,
    };

    let settings = LocalSettings {
        threads: 4,
        data_file_path: "../../data/batch.data",
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);
}
