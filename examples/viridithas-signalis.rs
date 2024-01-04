/// Network codenamed `signalis`, from Viridithas.
use bullet::{
    inputs, Activation, LocalSettings, LrScheduler, TrainerBuilder, TrainingSchedule, WdlScheduler,
};

fn main() {
    bullet::set_cbcs(true);

    #[rustfmt::skip]
    let mut trainer = TrainerBuilder::default()
        .set_batch_size(16_384)
        .set_eval_scale(400.0)
        .set_quantisations(&[181, 64])
        .set_input(
            inputs::ChessBucketsMirrored::new([
                0, 0, 0, 0,
                1, 1, 1, 1,
                2, 2, 2, 2,
                2, 2, 2, 2,
                3, 3, 3, 3,
                3, 3, 3, 3,
                3, 3, 3, 3,
                3, 3, 3, 3,
            ]
        ))
        .ft(1536)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let mut schedule = TrainingSchedule {
        net_id: "signalis".to_string(),
        start_epoch: 1,
        end_epoch: 15,
        wdl_scheduler: WdlScheduler::Constant { value: 0.4 },
        lr_scheduler: LrScheduler::Step {
            start: 0.001,
            gamma: 0.3,
            step: 4,
        },
        save_rate: 1,
    };

    let settings = LocalSettings {
        threads: 4,
        data_file_path: "thepile.data",
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);

    schedule.start_epoch = 16;
    schedule.end_epoch = 16;
    schedule.wdl_scheduler = WdlScheduler::Constant { value: 1.0 };

    trainer.run(&schedule, &settings);
}
