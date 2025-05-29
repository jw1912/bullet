use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, SparseInputType},
        outputs::MaterialCount,
    },
    nn::{optimiser::AdamW, InitSettings, Shape},
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader::DirectSequentialDataLoader, ValueTrainerBuilder},
};

const HL_SIZE: usize = 512;
const OUTPUT_BUCKETS: usize = 8;

fn main() {
    #[rustfmt::skip]
    let inputs = ChessBucketsMirrored::new([
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        2, 2, 2, 2,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
    ]);
    let num_inputs = inputs.num_inputs();

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
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .optimiser(AdamW)
        .save_format(&save_format)
        .build_custom(|builder, (stm, ntm, buckets), targets| {
            // factorise first layer weights
            let num_buckets = num_inputs / 768;
            let l0f = builder.new_weights("l0f", Shape::new(HL_SIZE * 768, 1), InitSettings::Zeroed);
            let ones = builder.new_constant(Shape::new(1, num_buckets), &vec![1.0; num_buckets]);
            let expanded_factoriser = l0f.matmul(ones).reshape(Shape::new(HL_SIZE, num_inputs));
            let mut l0 = builder.new_affine("l0", num_inputs, HL_SIZE);
            l0.weights = l0.weights + expanded_factoriser;

            // initialise layerstack
            let l1 = builder.new_affine("l1", HL_SIZE, OUTPUT_BUCKETS * 16);
            let l2 = builder.new_affine("l2", 30, OUTPUT_BUCKETS * 32);
            let l3 = builder.new_affine("l3", 32, OUTPUT_BUCKETS);

            // first layer inference
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let mut out = stm_subnet.concat(ntm_subnet);

            // layerstack inference
            out = l1.forward(out).select(buckets);

            let skip_neuron = out.slice_rows(15, 16);
            out = out.slice_rows(0, 15);

            out = out.concat(out.abs_pow(2.0));
            out = out.crelu();

            out = l2.forward(out).select(buckets).screlu();
            out = l3.forward(out).select(buckets);

            // network output
            out = out + skip_neuron;

            // squared error loss
            let loss = out.sigmoid().squared_error(targets);

            (out, loss)
        });

    let schedule = TrainingSchedule {
        net_id: "test".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 1024,
            start_superbatch: 1,
            end_superbatch: 10,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.3, step: 60 },
        save_rate: 150,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = DirectSequentialDataLoader::new(&["data/baseline.data"]);

    trainer.run(&schedule, &settings, &data_loader);

    let eval = 400.0 * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");
}
