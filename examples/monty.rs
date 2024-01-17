/*
The exact training used for monty's current network, updated as I merge new nets.
*/
use bullet::{
    inputs, Activation, LocalSettings, LrScheduler, TrainerBuilder, TrainingSchedule, WdlScheduler,
};

fn main() {
    let mut trainer = TrainerBuilder::default()
        .set_batch_size(16_384)
        .set_eval_scale(400.0)
        .set_input(inputs::ChessBucketsMirrored::new([0; 32]))
        .ft(1024)
        .activate(Activation::SCReLU)
        .add_layer(64)
        .activate(Activation::CReLU)
        .add_layer(32)
        .activate(Activation::ReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "monty-17.01.24".to_string(),
        start_epoch: 1,
        end_epoch: 5,
        wdl_scheduler: WdlScheduler::Linear {
            start: 0.0,
            end: 0.3,
        },
        lr_scheduler: LrScheduler::Step {
            start: 0.001,
            gamma: 0.1,
            step: 3,
        },
        save_rate: 1,
    };

    let settings = LocalSettings {
        threads: 4,
        data_file_path: "../../data/leela.data",
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);
    trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    trainer.eval("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 | 0 | 0.0");
    trainer.eval("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1 | 0 | 0.0");
    trainer.eval("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8 | 0 | 0.0");
    trainer.eval("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1 | 0 | 0.0");
}