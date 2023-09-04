use crate::{
    network::{
        Accumulator,
        Activation,
        NNUEParams,
        FEATURE_BIAS,
        OUTPUT_WEIGHTS,
        OUTPUT_BIAS,
        HIDDEN,
    },
    position::{Position, Features},
    util::sigmoid,
};

pub fn update_single_grad<Act: Activation>(
    pos: &Position,
    nnue: &NNUEParams,
    grad: &mut NNUEParams,
    error: &mut f32,
    blend: f32,
    scale: f32,
) {
    let bias = Accumulator::load_biases(nnue);
    let mut accs = [bias; 2];
    let mut activated = [[0.0; HIDDEN]; 2];
    let mut features = Features::default();

    let eval = nnue.forward::<Act>(pos, &mut accs, &mut activated, &mut features);

    let stm = pos.stm();
    let result = pos.blended_result(blend, stm, scale);

    let sigmoid = sigmoid(eval, 1.0);
    let err = (sigmoid - result) * sigmoid * (1. - sigmoid);
    *error += (sigmoid - result).powi(2);

    let mut components = [(0.0, 0.0); HIDDEN];

    for i in 0..HIDDEN {
        components[i] = (
            err * nnue[OUTPUT_WEIGHTS + i] * Act::activate_prime(accs[0][i]),
            err * nnue[OUTPUT_WEIGHTS + HIDDEN + i] * Act::activate_prime(accs[1][i]),
        );

        grad[FEATURE_BIAS + i] += components[i].0 + components[i].1;

        grad[OUTPUT_WEIGHTS + i] += err * activated[0][i];
        grad[OUTPUT_WEIGHTS + HIDDEN + i] += err * activated[1][i];
    }

    let opp = stm ^ 1;

    for (wfeat, bfeat) in features {
        let idxs = [wfeat * HIDDEN, bfeat * HIDDEN];
        let (widx, bidx) = (idxs[stm], idxs[opp]);
        for i in 0..HIDDEN {
            grad[widx + i] += components[i].0;
            grad[bidx + i] += components[i].1;
        }
    }

    grad[OUTPUT_BIAS] += err;
}