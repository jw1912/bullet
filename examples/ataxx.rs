use bullet_lib::{
    inputs, outputs, Activation, LocalSettings, LrScheduler, TrainerBuilder, TrainingSchedule,
    WdlScheduler,
};

const HIDDEN_SIZE: usize = 128;

fn main() {
    let mut trainer = TrainerBuilder::default()
        .single_perspective()
        .quantisations(&[255, 64])
        .input(inputs::Ataxx98)
        .output_buckets(outputs::Single)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "net005".to_string(),
        batch_size: 16_384,
        eval_scale: 400.0,
        batches_per_superbatch: 6104,
        start_superbatch: 1,
        end_superbatch: 40,
        wdl_scheduler: WdlScheduler::Constant { value: 0.5 },
        lr_scheduler: LrScheduler::Step {
            start: 0.001,
            gamma: 0.1,
            step: 15,
        },
        save_rate: 10,
    };

    let settings = LocalSettings {
        threads: 4,
        data_file_paths: vec!["../../data/ataxx/003.data"],
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);

    println!("{}", trainer.eval("x5o/7/7/7/7/7/o5x x 0 1"));
    println!("{}", trainer.eval("5oo/7/x6/x6/7/7/o5x o 0 2"));
}
