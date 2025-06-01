use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, SparseInputType},
        outputs::MaterialCount,
    },
    nn::{
        optimiser::{Ranger, RangerParams},
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

macro_rules! net_id {
    () => {
        "bullet_r63-768x8hm-1024-dp-pw-16-32-1x8"
    };
}

const NET_ID: &str = net_id!();

fn main() {
    #[rustfmt::skip]
    let inputs = ChessBucketsMirrored::new([
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        6, 6, 6, 6,
        7, 7, 7, 7,
        7, 7, 7, 7,
        7, 7, 7, 7,
        7, 7, 7, 7,
    ]);

    // network hyperparams
    let hl_size = 1024;
    let num_inputs = inputs.num_inputs();
    let num_buckets = num_inputs / 768;
    const NUM_OUTPUT_BUCKETS: usize = 8;

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
        .output_buckets(MaterialCount::<8>)
        .optimiser(Ranger)
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
            let l1 = builder.new_affine("l1", hl_size, NUM_OUTPUT_BUCKETS * 16);
            let l2 = builder.new_affine("l2", 16, NUM_OUTPUT_BUCKETS * 32);
            let l3 = builder.new_affine("l3", 32, NUM_OUTPUT_BUCKETS);

            // input layer inference
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let mut out = stm_subnet.concat(ntm_subnet);

            // add extra loss term to encourage sparsity
            let ones = builder.new_constant(Shape::new(1, hl_size), &vec![1.0; hl_size]);
            let nz_pen = 0.00001 * ones.matmul(out);

            // layerstack inference
            out = l1.forward(out).select(buckets).crelu();
            out = l2.forward(out).select(buckets).crelu();
            out = l3.forward(out).select(buckets);

            // squared error loss
            let loss = out.sigmoid().squared_error(targets) + nz_pen;
            (out, loss)
        });

    // cap l1 weights to 1.98 after factoriser is applied
    let l0_params = RangerParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };

    // allow float weights to have a large range
    let float_params = RangerParams { max_weight: 128.0, min_weight: -128.0, ..Default::default() };

    trainer.set_optimiser_params(RangerParams::default());
    trainer.optimiser_mut().set_params_for_weight("l0w", l0_params);
    trainer.optimiser_mut().set_params_for_weight("l0f", l0_params);
    trainer.optimiser_mut().set_params_for_weight("l2w", float_params);
    trainer.optimiser_mut().set_params_for_weight("l2b", float_params);
    trainer.optimiser_mut().set_params_for_weight("l3w", float_params);
    trainer.optimiser_mut().set_params_for_weight("l3b", float_params);

    let num_superbatches = 1000;
    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: 160.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: num_superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.3 },
        lr_scheduler: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.0, final_superbatch: num_superbatches },
        save_rate: 100,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 }; // 32
    let data_loader = DirectSequentialDataLoader::new(&["../../chess/data/rescored.data"]);

    //trainer.load_from_checkpoint("checkpoints/...");
    trainer.run(&schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/p1pp1pb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 160.0 * eval);
    }

    trainer.save_quantised(&format!("nets/{}.nn", NET_ID)).unwrap();
}
