use bullet_lib::{
    inputs, lr, optimiser, outputs, wdl, Activation, LocalSettings, Loss, TrainerBuilder, TrainingSchedule,
};

macro_rules! net_id {
    () => {
        "bullet_r36_768x8-1024x2-1x8"
    };
}

const NET_ID: &str = net_id!();

fn main() {
    #[rustfmt::skip]
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[255, 64])
        .optimiser(optimiser::AdamW)
        .input(inputs::ChessBucketsMirroredFactorised::new([
            0, 1, 2, 3,
            4, 4, 5, 5,
            6, 6, 6, 6,
            6, 6, 6, 6,
            7, 7, 7, 7,
            7, 7, 7, 7,
            7, 7, 7, 7,
            7, 7, 7, 7,
        ]))
        .output_buckets(outputs::MaterialCount::<8>)
        .feature_transformer(1024)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: 160.0,
        ft_regularisation: 0.0,
        batch_size: 16_384,
        batches_per_superbatch: 6104,
        start_superbatch: 1,
        end_superbatch: 400,
        wdl_scheduler: wdl::ConstantWDL { value: 0.3 },
        lr_scheduler: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.0, final_superbatch: 400 },
        loss_function: Loss::SigmoidMSE,
        save_rate: 10,
        optimiser_settings: optimiser::AdamWParams {
            decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            min_weight: -1.98,
            max_weight: 1.98,
        },
    };

    let settings = LocalSettings {
        threads: 12,
        data_file_paths: vec!["../../chess/data/shuffled.data"],
        test_set: None,
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);
}
