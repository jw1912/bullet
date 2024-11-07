/*
The exact training used for akimbo's current network, updated as I merge new nets.
*/
use bullet_lib::{
    inputs, loader, lr, optimiser, outputs,
    testing::{Engine, OpenBenchCompliant, OpeningBook, TestSettings, TimeControl, UciOption},
    wdl, Activation, LocalSettings, Loss, TrainerBuilder, TrainingSchedule, TrainingSteps,
};

const HIDDEN_SIZE: usize = 512;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;


fn main() {
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[QA, QB])
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMSE)
        .input(inputs::Chess768)
        .output_buckets(outputs::Single)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    trainer.load_from_checkpoint("C:\\NNUE-Trainer\\checkpoints\\simple-500\\");

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
