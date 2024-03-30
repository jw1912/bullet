/*
The exact training used for akimbo's current network, updated as I merge new nets.
*/
use bullet_lib::{
    inputs, outputs, Activation, Engine, LocalSettings, LrScheduler, OpeningBook, TestSettings, TimeControl,
    TrainerBuilder, TrainingSchedule, UciOption, WdlScheduler, Loss
};

macro_rules! net_id {
    () => {
        "lnet002"
    };
}

const NET_ID: &str = net_id!();

fn main() {
    #[rustfmt::skip]
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[255, 64])
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
        .output_buckets(outputs::Single)
        .feature_transformer(1024)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: 400.0,
        ft_regularisation: 0.0,
        batch_size: 16_384,
        batches_per_superbatch: 6104,
        start_superbatch: 1,
        end_superbatch: 240,
        wdl_scheduler: WdlScheduler::Constant { value: 0.0 },
        lr_scheduler: LrScheduler::Step { start: 0.001, gamma: 0.3, step: 60 },
        loss_function: Loss::SigmoidMSE,
        save_rate: 150,
    };

    let settings = LocalSettings {
        threads: 4,
        data_file_paths: vec!["../../data/test80-sep2022.data"],
        output_directory: "checkpoints",
    };

    let base_engine = Engine {
        repo: "https://github.com/jw1912/akimbo",
        branch: "main",
        bench: Some(2430757),
        net_path: None,
        uci_options: vec![UciOption("Hash", "16")],
    };

    let dev_engine = Engine {
        repo: "https://github.com/jw1912/akimbo",
        branch: "main",
        bench: None,
        net_path: None,
        uci_options: vec![UciOption("Hash", "16")],
    };

    let testing = TestSettings {
        test_rate: 20,
        out_dir: concat!("../../nets/", net_id!()),
        cutechess_path: "../../nets/cutechess-cli.exe",
        book_path: OpeningBook::Epd("../../nets/Pohl.epd"),
        num_game_pairs: 2000,
        concurrency: 6,
        time_control: TimeControl::FixedNodes(25_000),
        base_engine,
        dev_engine,
    };

    trainer.run_and_test(&schedule, &settings, &testing);

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
