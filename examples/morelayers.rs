/*
This is the result of some experimentation I did to try and train a network
for akimbo with more layers, it ended up being significantly stronger at
fixed-nodes, but unfortunately was too much of a slowdown to pass any
time-controlled test.
*/
use bullet::{
    inputs, Activation, LocalSettings, LrScheduler, TrainerBuilder, TrainingSchedule, WdlScheduler,
};

fn main() {
    let mut trainer = TrainerBuilder::default()
        .set_eval_scale(400.0)
        .set_input(inputs::Chess768)
        .ft(768)
        .activate(Activation::SCReLU)
        .add_layer(8)
        .activate(Activation::CReLU)
        .add_layer(16)
        .activate(Activation::CReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "morelayers".to_string(),
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
        data_file_path: "../../data/akimbo3-9.data",
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);
    trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    trainer.eval("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 | 0 | 0.0");
    trainer.eval("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1 | 0 | 0.0");
}
