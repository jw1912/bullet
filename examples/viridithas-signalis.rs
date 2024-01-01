/// Network codenamed `signalis`, from Viridithas.
///
/// NOTE: This network failed SPRT, but its a nice example for using buckets.
use bullet::{
    inputs, run_training, Activation, LrScheduler, LrSchedulerType, TrainerBuilder,
    TrainingSchedule, WdlScheduler,
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
        num_epochs: 15,
        wdl_scheduler: WdlScheduler::new(0.4, 0.4),
        lr_scheduler: LrScheduler::new(0.001, 0.3, LrSchedulerType::Step(4)),
        save_rate: 1,
    };

    run_training(
        &mut trainer,
        &mut schedule,
        8,
        "../../thepile.data",
        "checkpoints",
    );
}
