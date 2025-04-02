/*
This is used to confirm non-functional changes for bullet.
*/
use bullet_lib::{
    nn::{optimiser, Activation, GraphCompileArgs},
    trainer::{
        default::{inputs, loader, outputs, Loss, TrainerBuilder},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
};

fn main() {
    let mut trainer = TrainerBuilder::default()
        .set_compile_args(GraphCompileArgs::default().emit_ir())
        .quantisations(&[181, 64])
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMSE)
        .input(inputs::Chess768)
        .output_buckets(outputs::Single)
        .feature_transformer(32)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    trainer.load_from_checkpoint("checkpoints/testnet");

    let schedule = TrainingSchedule {
        net_id: "testnet".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps { batch_size: 16_384, batches_per_superbatch: 1, start_superbatch: 1, end_superbatch: 10 },
        wdl_scheduler: wdl::ConstantWDL { value: 0.2 },
        lr_scheduler: lr::ConstantLR { value: 0.001 },
        save_rate: 10,
    };

    trainer.set_optimiser_params(optimiser::AdamWParams::default());

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["data/batch1.data"]);
    trainer.profile_all_nodes();
    trainer.run(&schedule, &settings, &data_loader);
    trainer.report_profiles();
    trainer.sanity_check();
}
