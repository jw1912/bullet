mod accumulator;
mod nnue;
mod quantise;

pub use accumulator::Accumulator;
pub use nnue::{NNUEParams, HIDDEN, K};
pub use quantise::QuantisedNNUE;

use data::Position;

fn activate(x: f64) -> f64 {
    x.max(0.0)
}

fn activate_prime(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        1.0
    }
}

pub fn update_single_grad(
    pos: &Position,
    nnue: &NNUEParams,
    grad: &mut NNUEParams,
    error: &mut f64,
) {
    let mut accs = [Accumulator::new(nnue.feature_bias); 2];
    let mut features = [(0, 0); 32];
    let mut len = 0;

    for (piece, square) in pos.into_iter() {
        let wfeat = 64 * piece as usize + square as usize;
        let bfeat = 64 * ((piece as usize + 6) % 12) + (56 ^ (square as usize));
        features[len] = (wfeat, bfeat);
        len += 1;
        accs[0].add_feature(wfeat, nnue);
        accs[1].add_feature(bfeat, nnue);
    }

    let mut eval = nnue.output_bias;
    let mut activated = [Accumulator::new([0.0; HIDDEN]); 2];

    let side = usize::from(pos.stm);
    let (boys, opps) = (&accs[side], &accs[side ^ 1]);

    for (idx, (&i, &w)) in boys.iter().zip(&nnue.output_weights[..HIDDEN]).enumerate() {
        activated[0][idx] = activate(i);
        eval += activated[0][idx] * w;
    }

    for (idx, (&i, &w)) in opps.iter().zip(&nnue.output_weights[HIDDEN..]).enumerate() {
        activated[1][idx] = activate(i);
        eval += activated[1][idx] * w;
    }

    let result = f64::from(pos.result + 1) / 2.;

    let sigmoid = 1. / (1. + (-eval * K).exp());
    let err = (sigmoid - result) * sigmoid * (1. - sigmoid);
    *error += (sigmoid - result).powi(2);

    let mut components = Accumulator::new([(0.0, 0.0); HIDDEN]);

    for i in 0..HIDDEN {
        components[i] = (
            err * nnue.output_weights[i] * activate_prime(boys[i]),
            err * nnue.output_weights[HIDDEN + i] * activate_prime(opps[i])
        );

        grad.feature_bias[i] += components[i].0 + components[i].1;

        grad.output_weights[i] += err * activated[0][i];
        grad.output_weights[HIDDEN + i] += err * activated[1][i];
    }

    for (wfeat, bfeat) in features.iter().take(len) {
        for i in 0..HIDDEN {
            grad.feature_weights[wfeat * HIDDEN + i] += components[i].0;
            grad.feature_weights[bfeat * HIDDEN + i] += components[i].1;
        }
    }

    grad.output_bias += err;
}
