#![allow(deprecated)]
use bullet_lib::{
    nn::{optimiser, Activation},
    trainer::{
        default::{
            inputs, loader,
            testing::{Engine, GameRunnerPath, OpenBenchCompliant, OpeningBook, TestSettings, TimeControl, UciOption},
            Loss, TrainerBuilder,
        },
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
};

const NET_ID: &str = "lnet002";

fn main() {
    #[rustfmt::skip]
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[255, 64])
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMSE)
        .input(inputs::ChessBucketsMirrored::new([
            0, 0, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 3, 3, 3,
        ]))
        .feature_transformer(1024)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 240,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.3, step: 60 },
        save_rate: 150,
    };

    trainer.set_optimiser_params(optimiser::AdamWParams::default());

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["data/baseline.data"]);

    let engine = |bench| Engine {
        repo: "https://github.com/jw1912/akimbo",
        branch: "main",
        bench,
        net_path: None,
        uci_options: vec![UciOption("Hash", "16")],
        engine_type: OpenBenchCompliant,
    };

    let testing = TestSettings {
        test_rate: 20,
        out_dir: &format!("../../nets/{NET_ID}"),
        gamerunner_path: GameRunnerPath::CuteChess("../../nets/cutechess-cli.exe"),
        book_path: OpeningBook::Epd("../../nets/UHO_Lichess_4852_v1.epd"),
        num_game_pairs: 2000,
        concurrency: 6,
        time_control: TimeControl::FixedNodes(25_000),
        base_engine: engine(Some(2256851)),
        dev_engine: engine(None),
    };

    trainer.run_and_test(&schedule, &settings, &data_loader, &testing);

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
