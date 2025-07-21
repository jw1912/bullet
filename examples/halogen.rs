use bullet_lib::{
    game::{
        inputs::{get_num_buckets, ChessBucketsMirrored},
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
    value::{
        loader::{viribinpack, ViriBinpackLoader},
        ValueTrainerBuilder,
    },
};

macro_rules! net_id {
    () => {
        "bullet_r81-768x8hm-1536-dp-pw-16-da-32-1x8"
    };
}

const NET_ID: &str = net_id!();

fn main() {
    // network hyperparams
    let ft_size = 1536;
    let l1_size = 16;
    let l2_size = 32;
    const NUM_OUTPUT_BUCKETS: usize = 8;
    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        6, 6, 6, 6,
        7, 7, 7, 7,
        7, 7, 7, 7,
        7, 7, 7, 7,
        7, 7, 7, 7,
    ];
    const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

    let save_format = [
        SavedFormat::id("l0w")
            .add_transform(|builder, _, mut weights| {
                let factoriser = builder.get_weights("l0f").get_dense_vals().unwrap();
                let expanded = factoriser.repeat(NUM_INPUT_BUCKETS);
                for (i, &j) in weights.iter_mut().zip(expanded.iter()) {
                    *i += j;
                }
                weights
            })
            .quantise::<i16>(255)
            .round(),
        SavedFormat::id("l0b").quantise::<i16>(255).round(),
        SavedFormat::id("l1w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("l1b").quantise::<i16>(64 * 255).round(),
        SavedFormat::id("l2w").transpose().round(),
        SavedFormat::id("l2b").round(),
        SavedFormat::id("l3w").transpose().round(),
        SavedFormat::id("l3b").round(),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<8>)
        .optimiser(Ranger)
        .save_format(&save_format)
        .build_custom(|builder, (stm, ntm, buckets), targets| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(ft_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, ft_size);
            l0.weights = l0.weights + expanded_factoriser;

            // layerstack weights
            let l1 = builder.new_affine("l1", ft_size, NUM_OUTPUT_BUCKETS * l1_size);
            let l2 = builder.new_affine("l2", l1_size * 2, NUM_OUTPUT_BUCKETS * l2_size);
            let l3 = builder.new_affine("l3", l2_size, NUM_OUTPUT_BUCKETS);

            // input layer inference
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let mut out = stm_subnet.concat(ntm_subnet);

            // layerstack inference
            out = l1.forward(out).select(buckets);
            out = out.concat(out.abs_pow(2.0)).crelu();
            out = l2.forward(out).select(buckets).crelu();
            out = l3.forward(out).select(buckets);

            // squared error loss
            let loss = out.sigmoid().squared_error(targets);
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
        wdl_scheduler: wdl::ConstantWDL { value: 0.5 },
        lr_scheduler: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.0, final_superbatch: num_superbatches },
        save_rate: 100,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };
    let data_loader = ViriBinpackLoader::new(
        "..\\..\\chess\\data\\training.viri",
        1024 * 32,
        4,
        viribinpack::ViriFilter::Builtin(viriformat::dataformat::Filter {
            min_ply: 0,
            min_pieces: 0,
            ..Default::default()
        }),
    );

    //trainer.load_from_checkpoint(&format!("checkpoints/{NET_ID}-"));
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

    trainer.save_quantised(&format!("nets/{NET_ID}-e{num_superbatches}.nn")).unwrap();
}
