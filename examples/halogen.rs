use bullet_lib::{
    inputs, lr, optimiser, outputs, wdl, Activation, LocalSettings, Loss, TrainerBuilder, TrainingSchedule,
};

macro_rules! net_id {
    () => {
        "bullet_r23_768x4x2-768x2-1x8"
    };
}

const NET_ID: &str = net_id!();

fn main() {
    #[rustfmt::skip]
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[181, 64])
        .optimiser(optimiser::AdamW)
        .input(inputs::ChessBucketsMirrored::new([
            0, 0, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 3, 3, 3,
        ]))
        .output_buckets(outputs::MaterialCount::<8>)
        .feature_transformer(768)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: 160.0,
        ft_regularisation: 0.0,
        batch_size: 16_384,
        batches_per_superbatch: 68128,
        start_superbatch: 1,
        end_superbatch: 50,
        wdl_scheduler: wdl::ConstantWDL { value: 0.3 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.95, step: 1 },
        loss_function: Loss::SigmoidMSE,
        save_rate: 10,
        optimiser_settings: optimiser::AdamWParams { decay: 0.01 },
    };

    let settings = LocalSettings {
        threads: 12,
        data_file_paths: vec!["../../chess/data/shuffled.data"],
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);
}
