mod accumulator;
mod nnue;
mod quantise;

pub use accumulator::Accumulator;
pub use nnue::{NNUEParams, HIDDEN, K};
pub use quantise::QuantisedNNUE;

use crate::position::Position;

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
    let mut acc = Accumulator::new(nnue.feature_bias);

    for &feature in pos.active.iter().take(pos.num) {
        acc.add_feature(usize::from(feature), nnue);
    }

    let mut eval = nnue.output_bias;
    let mut act = Accumulator::new([0.0; HIDDEN]);
    for (idx, (&i, &w)) in acc.iter().zip(&nnue.output_weights).enumerate() {
        act[idx] = activate(i);
        eval += act[idx] * w;
    }

    let sigmoid = 1. / (1. + (-eval * K).exp());
    let err = (sigmoid - pos.result) * sigmoid * (1. - sigmoid);
    *error += (sigmoid - pos.result).powi(2);

    for i in 0..HIDDEN {
        let component = err * nnue.output_weights[i] * activate_prime(acc[i]);

        for &j in pos.active.iter().take(pos.num) {
            grad.feature_weights[usize::from(j) * HIDDEN + i] += component;
        }

        grad.feature_bias[i] += component;

        grad.output_weights[i] += err * act[i];
    }

    grad.output_bias += err;
}
