mod accumulator;
mod nnue;
mod quantise;

pub use accumulator::Accumulator;
pub use nnue::{NNUEParams, HIDDEN, INPUT, K};
pub use quantise::QuantisedNNUE;

use crate::activation::Activation;
use data::Position;

pub fn update_single_grad<Act: Activation>(
    pos: &Position,
    nnue: &NNUEParams,
    grad: &mut NNUEParams,
    error: &mut f32,
    blend: f32,
) {
    let mut accs = [Accumulator::new(nnue.feature_bias); 2];
    let mut features = [(0, 0); 32];
    let mut len = 0;

    let stm = pos.stm();

    for (colour, piece, square) in pos.into_iter() {
        let c = usize::from(colour);
        let pc = usize::from(piece);
        let sq = usize::from(square);
        let wfeat = [0, 384][c] + 64 * pc + sq;
        let bfeat = [384, 0][c] + 64 * pc + (sq ^ 56);

        features[len] = (wfeat, bfeat);
        len += 1;
        accs[stm].add_feature(wfeat, nnue);
        accs[stm ^ 1].add_feature(bfeat, nnue);
    }

    let mut eval = nnue.output_bias;
    let mut activated = [Accumulator::new([0.0; HIDDEN]); 2];

    for (idx, (&i, &w)) in accs[0]
        .iter()
        .zip(&nnue.output_weights[..HIDDEN])
        .enumerate()
    {
        activated[0][idx] = Act::activate(i);
        eval += activated[0][idx] * w;
    }

    for (idx, (&i, &w)) in accs[1]
        .iter()
        .zip(&nnue.output_weights[HIDDEN..])
        .enumerate()
    {
        activated[1][idx] = Act::activate(i);
        eval += activated[1][idx] * w;
    }

    let result = pos.blended_result(blend, stm);

    let sigmoid = data::util::sigmoid(eval, K);
    let err = (sigmoid - result) * sigmoid * (1. - sigmoid);
    *error += (sigmoid - result).powi(2);

    let mut components = Accumulator::new([(0.0, 0.0); HIDDEN]);

    for i in 0..HIDDEN {
        components[i] = (
            err * nnue.output_weights[i] * Act::activate_prime(accs[0][i]),
            err * nnue.output_weights[HIDDEN + i] * Act::activate_prime(accs[1][i]),
        );

        grad.feature_bias[i] += components[i].0 + components[i].1;

        grad.output_weights[i] += err * activated[0][i];
        grad.output_weights[HIDDEN + i] += err * activated[1][i];
    }

    for (wfeat, bfeat) in features.iter().take(len) {
        let idxs = [wfeat * HIDDEN, bfeat * HIDDEN];
        let (widx, bidx) = (idxs[stm], idxs[stm ^ 1]);
        for i in 0..HIDDEN {
            grad.feature_weights[widx + i] += components[i].0;
            grad.feature_weights[bidx + i] += components[i].1;
        }
    }

    grad.output_bias += err;
}

fn eval<Act: Activation>(pos: &Position, nnue: &NNUEParams) -> f32 {
    let mut accs = [Accumulator::new(nnue.feature_bias); 2];

    for (colour, piece, square) in pos.into_iter() {
        let c = usize::from(colour);
        let pc = usize::from(piece);
        let sq = usize::from(square);
        let wfeat = [0, 384][c] + 64 * pc + sq;
        let bfeat = [384, 0][c] + 64 * pc + (sq ^ 56);
        accs[0].add_feature(wfeat, nnue);
        accs[1].add_feature(bfeat, nnue);
    }

    let mut eval = nnue.output_bias;

    let side = pos.stm();
    let (boys, opps) = (&accs[side], &accs[side ^ 1]);

    for (&i, &w) in boys.iter().zip(&nnue.output_weights[..HIDDEN]) {
        eval += Act::activate(i) * w;
    }

    for (&i, &w) in opps.iter().zip(&nnue.output_weights[HIDDEN..]) {
        eval += Act::activate(i) * w;
    }

    eval * 400.
}

pub fn test_eval<Act: Activation>(nnue: &NNUEParams) {
    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 0 [0.5]",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 0 [0.0]",
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1 0 [1.0]",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1 0 [1.0]",
    ];

    println!("\n===Test Positions===");

    for fen in fens {
        let pos = Position::from_epd(fen).unwrap();
        println!("FEN: {fen}");
        println!("EVAL: {}", eval::<Act>(&pos, nnue));
        println!();
    }
}
