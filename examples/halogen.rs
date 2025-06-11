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
    },
    value::{
        loader::{viribinpack, ViriBinpackLoader},
        ValueTrainerBuilder,
    },
};

macro_rules! net_id {
    () => {
        "bullet_r69-768x8hm-1024-dp-1x8"
    };
}

const NET_ID: &str = net_id!();

fn main() {
    // network hyperparams
    let hl_size = 1024;
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
            .quantise::<i16>(255),
        SavedFormat::id("l0b").quantise::<i16>(255),
        SavedFormat::id("l1w").quantise::<i16>(64).transpose(),
        SavedFormat::id("l1b").quantise::<i16>(64 * 255),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<8>)
        .optimiser(Ranger)
        .save_format(&save_format)
        .build_custom(|builder, (stm, ntm, buckets), targets| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, hl_size);
            l0.weights = l0.weights + expanded_factoriser;

            // output layer weights
            let l1 = builder.new_affine("l1", 2 * hl_size, NUM_OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm).screlu();
            let ntm_hidden = l0.forward(ntm).screlu();
            let mut out = stm_hidden.concat(ntm_hidden);
            out = l1.forward(out).select(buckets);

            // squared error loss
            let loss = out.sigmoid().squared_error(targets);
            (out, loss)
        });

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

    let settings = LocalSettings { threads: 32, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let data_loader = ViriBinpackLoader::new(
        "../../chess/data/training.viri",
        1024 * 48,
        32,
        viribinpack::ViriFilter::Builtin(viriformat::dataformat::Filter {
            min_ply: 0,
            min_pieces: 0,
            ..Default::default()
        }),
    );

    trainer.set_optimiser_params(RangerParams::default());
    //trainer.load_from_checkpoint(&format!("checkpoints/{NET_ID}-{num_superbatches}"));
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
