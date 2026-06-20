use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{InitSettings, Shape, optimiser::AdamW},
    trainer::{
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

fn main() {
    #[rustfmt::skip]
    let bucket_layout = [
        0, 0, 0, 0,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
    ];

    let hl_size = 128;
    let num_input_buckets = get_num_buckets(&bucket_layout);
    let num_inputs = 768 * num_input_buckets;
    let num_output_buckets = 8;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(bucket_layout))
        .output_buckets(MaterialCount::<8>)
        .save_format(&[])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(num_input_buckets);

            let mut l0 = builder.new_affine("l0", num_inputs, hl_size);
            l0.weights = l0.weights + expanded_factoriser;

            let l1 = builder.new_affine("l1", hl_size, num_output_buckets * 16);
            let l2 = builder.new_affine("l2", 30, num_output_buckets * 32);
            let l3 = builder.new_affine("l3", 32, num_output_buckets);

            let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
            let stm_subnet = ft(stm_inputs, 0, hl_size / 2) * ft(stm_inputs, hl_size / 2, hl_size);
            let ntm_subnet = ft(ntm_inputs, 0, hl_size / 2) * ft(ntm_inputs, hl_size / 2, hl_size);
            let mut out = stm_subnet.concat(ntm_subnet);

            out = l1.forward(out).select(buckets);

            let skip_neuron = out.slice_rows(15, 16);

            out = out.slice_rows(0, 15);
            out = out.concat(out * out).crelu();

            out = l2.forward(out).select(buckets).screlu();
            out = l3.forward(out).select(buckets);

            out + skip_neuron
        });

    let schedule = TrainingSchedule {
        net_id: "test2".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps { batch_size: 16_384, batches_per_superbatch: 1, start_superbatch: 1, end_superbatch: 10 },
        wdl_scheduler: wdl::LinearWDL { start: 0.2, end: 0.0 },
        lr_scheduler: lr::DropLR { start: 0.001, gamma: 0.1, drop: 5 },
        save_rate: 10,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = DirectSequentialDataLoader::new(&["examples/tests/batch.bf"]);

    let eval = 400.0 * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");

    trainer.run(&schedule, &settings, &data_loader);

    println!("Expected loss: 0.052189");

    let eval = 400.0 * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");
}
