/*
This is the result of some experimentation I did to try and train a network
for akimbo with more layers, it ended up being significantly stronger at
fixed-nodes, but unfortunately was too much of a slowdown to pass any
time-controlled test.
*/
use bullet_lib::{
    inputs, optimiser, outputs, Activation, LocalSettings, Loss, LrScheduler, TrainerBuilder, TrainingSchedule,
    WdlScheduler,
};

fn main() {
    let mut trainer = TrainerBuilder::default()
        .optimiser(optimiser::AdamW)
        .input(inputs::Chess768)
        .output_buckets(outputs::Single)
        .feature_transformer(768)
        .activate(Activation::SCReLU)
        .add_layer(8)
        .activate(Activation::CReLU)
        .add_layer(16)
        .activate(Activation::CReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "morelayers".to_string(),
        eval_scale: 400.0,
        ft_regularisation: 0.0,
        batch_size: 16_384,
        batches_per_superbatch: 6104,
        start_superbatch: 1,
        end_superbatch: 255,
        wdl_scheduler: WdlScheduler::Linear { start: 0.2, end: 0.5 },
        lr_scheduler: LrScheduler::Step { start: 0.001, gamma: 0.1, step: 120 },
        loss_function: Loss::SigmoidMSE,
        save_rate: 1,
        optimiser_settings: optimiser::AdamWParams { decay: 0.01 },
    };

    let settings = LocalSettings {
        threads: 4,
        data_file_paths: vec!["../../data/akimbo3-9.data"],
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);
}
