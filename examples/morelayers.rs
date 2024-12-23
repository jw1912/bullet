/*
This is the result of some experimentation I did to try and train a network
for akimbo with more layers, it ended up being significantly stronger at
fixed-nodes, but unfortunately was too much of a slowdown to pass any
time-controlled test.
*/
use bullet_lib::{
    default::{inputs, loader, outputs, Loss, TrainerBuilder},
    lr, optimiser, wdl, Activation, LocalSettings, TrainingSchedule, TrainingSteps,
};

fn main() {
    let mut trainer = TrainerBuilder::default()
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMSE)
        .input(inputs::Chess768)
        .output_buckets(outputs::MaterialCount::<8>)
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
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 255,
        },
        wdl_scheduler: wdl::LinearWDL { start: 0.2, end: 0.5 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 120 },
        save_rate: 1,
    };

    trainer.set_optimiser_params(optimiser::AdamWParams::default());

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["data/baseline.data"]);

    trainer.run(&schedule, &settings, &data_loader);
}
