use bullet::{
    inputs, outputs, Activation, LocalSettings, LrScheduler, TrainerBuilder, TrainingSchedule, WdlScheduler,
};

const HIDDEN_SIZE: usize = 2;

fn main() {
    let mut trainer = TrainerBuilder::default()
        .single_perspective()
        .input(inputs::Ataxx147)
        .output_buckets(outputs::Single)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "net001".to_string(),
        batch_size: 16_384,
        eval_scale: 400.0,
        start_epoch: 1,
        end_epoch: 10,
        wdl_scheduler: WdlScheduler::Constant { value: 1.0 },
        lr_scheduler: LrScheduler::Step {
            start: 0.001,
            gamma: 0.1,
            step: 6,
        },
        save_rate: 1,
    };

    let settings = LocalSettings {
        threads: 4,
        data_file_paths: vec!["../../data/ataxx/1.data"],
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);

    println!("{}", trainer.eval("x5o/7/7/7/7/7/o5x x 0 1"));
    println!("{}", trainer.eval("5oo/7/x6/x6/7/7/o5x o 0 2"));
}