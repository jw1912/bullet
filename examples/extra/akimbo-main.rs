/*
The exact training used for akimbo's current network, updated as I merge new nets.
*/
use bullet_lib::{
    game::{
        formats::sfbinpack::{
            chess::{piecetype::PieceType, r#move::MoveType},
            TrainingDataEntry,
        },
        inputs::SparseInputType,
    },
    nn::optimiser::AdamW,
    trainer::{
        default::inputs::ChessBucketsMirrored,
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader::SfBinpackLoader, ValueTrainerBuilder},
};

const NET_ID: &str = "lnet003";
const DATA_PATH: &str = "data/test80-2024-02-feb-2tb7p.min-v2.v6.binpack";

fn main() {
    #[rustfmt::skip]
    let inputs = ChessBucketsMirrored::new([
        0, 0, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
    ]);

    let fmt = [
        SavedFormat::id("l0w").quantise::<i16>(255),
        SavedFormat::id("l0b").quantise::<i16>(255),
        SavedFormat::id("l1w").quantise::<i8>(64),
        SavedFormat::id("l1b").quantise::<i32>(64 * 255),
        SavedFormat::id("l2w").quantise::<i32>(255),
        SavedFormat::id("l2b").quantise::<i32>(255),
        SavedFormat::id("l3w").quantise::<i32>(255),
        SavedFormat::id("l3b").quantise::<i32>(255 * 255),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(inputs)
        .save_format(&fmt)
        .loss_fn(|output, targets| output.sigmoid().squared_error(targets))
        .build(|builder, stm, ntm| {
            let l0 = builder.new_affine("l0", inputs.num_inputs(), 1024);
            let l1 = builder.new_affine("l1", 1024, 16);
            let l2 = builder.new_affine("l2", 16, 32);
            let l3 = builder.new_affine("l3", 32, 1);

            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let hl1 = stm_subnet.concat(ntm_subnet);
            let hl2 = l1.forward(hl1).screlu();
            let hl3 = l2.forward(hl2).screlu();
            l3.forward(hl3)
        });

    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 480,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::Warmup { inner: lr::StepLR { start: 0.001, gamma: 0.1, step: 180 }, warmup_batches: 200 },
        save_rate: 150,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };

    let data_loader = SfBinpackLoader::new(DATA_PATH, 4096, 4, filter);

    trainer.run(&schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * eval);
    }
}

fn filter(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && entry.score.unsigned_abs() <= 10000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
        && !entry.pos.is_checked(entry.pos.side_to_move())
}
