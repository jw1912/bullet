/*
The exact training used for akimbo's current network, updated as I merge new nets.
*/
use bullet::{
    inputs, outputs, Activation, LocalSettings, LrScheduler, TrainerBuilder, TrainingSchedule, WdlScheduler,
};

fn main() {
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[181, 64])
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
        .feature_transformer(768)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "net-07.02.24".to_string(),
        batch_size: 16_384,
        eval_scale: 400.0,
        start_epoch: 1,
        end_epoch: 17,
        wdl_scheduler: WdlScheduler::Linear {
            start: 0.2,
            end: 0.5,
        },
        lr_scheduler: LrScheduler::Step {
            start: 0.001,
            gamma: 0.1,
            step: 8,
        },
        save_rate: 1,
    };

    let settings = LocalSettings {
        threads: 4,
        data_file_paths: vec!["../../data/akimbo3-8.data"],
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1 | 0 | 0.0",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * eval);
    }
}
