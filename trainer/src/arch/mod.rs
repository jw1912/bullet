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
    let mut accumulator = Accumulator::new(nnue.feature_bias);

    for (piece, square) in pos.into_iter() {
        let feature = 64 * piece as usize + square as usize;
        accumulator.add_feature(feature, nnue);
    }

    let mut eval = nnue.output_bias;
    let mut activated = Accumulator::new([0.0; HIDDEN]);
    for (idx, (&i, &w)) in accumulator.iter().zip(&nnue.output_weights).enumerate() {
        activated[idx] = activate(i);
        eval += activated[idx] * w;
    }

    let sigmoid = 1. / (1. + (-eval * K).exp());
    let result = f64::from(pos.result + 1) / 2.;
    let err = (sigmoid - result) * sigmoid * (1. - sigmoid);
    *error += (sigmoid - result).powi(2);

    let mut components = Accumulator::new([0.0; HIDDEN]);

    for i in 0..HIDDEN {
        components[i] = err * nnue.output_weights[i] * activate_prime(accumulator[i]);

        grad.feature_bias[i] += components[i];

        grad.output_weights[i] += err * activated[i];
    }

    for (piece, square) in pos.into_iter() {
        let feature = 64 * piece as usize + square as usize;
        for i in 0..HIDDEN {
            grad.feature_weights[feature * HIDDEN + i] += components[i];
        }
    }

    grad.output_bias += err;
}
