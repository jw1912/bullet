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
    position::Position,
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
    let mut features = [(0, 0); 32];
    let mut len = 0;

    let stm = pos.stm();
    let opp = stm ^ 1;

    for (colour, piece, square) in pos.into_iter() {
        let c = usize::from(colour);
        let pc = 64 * usize::from(piece);
        let sq = usize::from(square);
        let wfeat = [0, 384][c] + pc + sq;
        let bfeat = [384, 0][c] + pc + (sq ^ 56);

        features[len] = (wfeat, bfeat);
        len += 1;
        accs[stm].add_feature(wfeat, nnue);
        accs[opp].add_feature(bfeat, nnue);
    }

    let mut eval = nnue[OUTPUT_BIAS];
    let mut activated = [[0.0; HIDDEN]; 2];

    for (idx, (&i, &w)) in accs[0]
        .iter()
        .zip(&nnue[OUTPUT_WEIGHTS..OUTPUT_WEIGHTS + HIDDEN])
        .enumerate()
    {
        activated[0][idx] = Act::activate(i);
        eval += activated[0][idx] * w;
    }

    for (idx, (&i, &w)) in accs[1]
        .iter()
        .zip(&nnue[OUTPUT_WEIGHTS + HIDDEN..OUTPUT_BIAS])
        .enumerate()
    {
        activated[1][idx] = Act::activate(i);
        eval += activated[1][idx] * w;
    }

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

    for (wfeat, bfeat) in features.iter().take(len) {
        let idxs = [wfeat * HIDDEN, bfeat * HIDDEN];
        let (widx, bidx) = (idxs[stm], idxs[opp]);
        for i in 0..HIDDEN {
            grad[widx + i] += components[i].0;
            grad[bidx + i] += components[i].1;
        }
    }

    grad[OUTPUT_BIAS] += err;
}