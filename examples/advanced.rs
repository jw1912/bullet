use bullet_lib::{
    game::{
        formats::bulletformat::ChessBoard,
        inputs::{ChessBucketsMirrored, SparseInputType},
        outputs::OutputBuckets,
    },
    nn::{
        optimiser::{AdamW, AdamWParams},
        InitSettings, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
        NetworkTrainer,
    },
    value::{loader::DirectSequentialDataLoader, ValueTrainerBuilder},
};

const NUM_OUTPUT_BUCKETS: usize = 8;

#[derive(Clone, Copy, Default)]
pub struct CustomOutputBuckets;
impl OutputBuckets<ChessBoard> for CustomOutputBuckets {
    const BUCKETS: usize = NUM_OUTPUT_BUCKETS;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let divisor = 32usize.div_ceil(NUM_OUTPUT_BUCKETS);
        let piece_count = pos.occ().count_ones() as u8;
        (piece_count - 2) / divisor as u8
    }
}

fn main() {
    // king-bucketed inputs
    #[rustfmt::skip]
    let inputs = ChessBucketsMirrored::new([
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7,
        8, 8, 8, 8,
        8, 8, 8, 8,
        9, 9, 9, 9,
        9, 9, 9, 9,
    ]);

    // network hyperparams
    let hl_size = 512;
    let num_inputs = inputs.num_inputs();
    let num_buckets = num_inputs / 768;
    let num_output_buckets = NUM_OUTPUT_BUCKETS;

    assert!(num_buckets > 1, "Factoriser is worthless with only one bucket!");
    assert!(num_output_buckets > 1, "1 output bucket does not make sense!");

    let save_format = [
        // factoriser weights need to be merged
        SavedFormat::id("l0f").quantise::<i16>(255),
        SavedFormat::id("l0w").quantise::<i16>(255),
        SavedFormat::id("l0b").quantise::<i16>(255),
        SavedFormat::id("l1w").quantise::<i16>(64).transpose(),
        SavedFormat::id("l1b").quantise::<i16>(64 * 255),
        SavedFormat::id("l2w").transpose(),
        SavedFormat::id("l2b"),
        SavedFormat::id("l3w").transpose(),
        SavedFormat::id("l3b"),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(inputs)
        .output_buckets(CustomOutputBuckets)
        .optimiser(AdamW)
        .save_format(&save_format)
        .build_custom(|builder, (stm, ntm, buckets), targets| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(hl_size * 768, 1), InitSettings::Zeroed);
            let ones = builder.new_constant(Shape::new(1, num_buckets), &vec![1.0; num_buckets]);
            let expanded_factoriser = l0f.matmul(ones).reshape(Shape::new(hl_size, num_inputs));

            // input layer weights
            let mut l0 = builder.new_affine("l0", num_inputs, hl_size);
            l0.weights = l0.weights + expanded_factoriser;

            // layerstack weights
            let l1 = builder.new_affine("l1", hl_size, num_output_buckets * 16);
            let l2 = builder.new_affine("l2", 30, num_output_buckets * 32);
            let l3 = builder.new_affine("l3", 32, num_output_buckets);

            // input layer inference
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let mut out = stm_subnet.concat(ntm_subnet);

            // layerstack inference
            out = l1.forward(out).select(buckets);

            let skip_neuron = out.slice_rows(15, 16);

            out = out.slice_rows(0, 15);
            out = out.concat(out.abs_pow(2.0)).crelu();

            out = l2.forward(out).select(buckets).screlu();
            out = l3.forward(out).select(buckets);

            // network output
            out = out + skip_neuron;

            // squared error loss
            let loss = out.sigmoid().squared_error(targets);

            (out, loss)
        });

    // need to account for factoriser weight magnitudes
    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser_mut().set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser_mut().set_params_for_weight("l0f", stricter_clipping);

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
