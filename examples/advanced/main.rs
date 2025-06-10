pub mod output_buckets;
pub mod trainer;

use bullet_lib::{
    trainer::{
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::loader::DirectSequentialDataLoader,
};

fn main() {
    // king-bucketed inputs
    #[rustfmt::skip]
    let bucket_layout = [
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7,
        8, 8, 8, 8,
        8, 8, 8, 8,
        9, 9, 9, 9,
        9, 9, 9, 9,
    ];

    let hl_size = 1024;

    let mut trainer = trainer::make_trainer(bucket_layout, hl_size);

    let num_superbatches = 800;
    let schedule = TrainingSchedule {
        net_id: "test".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: num_superbatches,
        },
        wdl_scheduler: wdl::LinearWDL { start: 0.0, end: 0.5 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 0.001,
            final_lr: 0.001 * 0.3f32.powi(5),
            final_superbatch: num_superbatches,
        },
        save_rate: 150,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let data_loader = DirectSequentialDataLoader::new(&["data/baseline.data"]);

    trainer.run(&schedule, &settings, &data_loader);

    let eval = 400.0 * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");
}
