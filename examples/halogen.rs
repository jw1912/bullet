use bullet_lib::{
    nn::{optimiser, Activation},
    trainer::{
        default::{inputs, loader, outputs, Loss, TrainerBuilder},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
};

macro_rules! net_id {
    () => {
        "bullet_r43_768x8-1024x2-1x8"
    };
}

const NET_ID: &str = net_id!();

fn main() {
    #[rustfmt::skip]
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[255, 64])
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMSE)
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
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 400,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.3 },
        lr_scheduler: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.0, final_superbatch: 400 },
        save_rate: 10,
    };

    let settings =
        LocalSettings { threads: 12, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["../../chess/data/shuffled.data"]);

    trainer.set_optimiser_params(optimiser::AdamWParams::default());
    trainer.run(&schedule, &settings, &data_loader);
}
