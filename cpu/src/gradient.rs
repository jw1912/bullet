use super::{
    Accumulator,
    NetworkParams,
    FEATURE_BIAS,
    OUTPUT_BIAS,
    OUTPUT_WEIGHTS,
    InputType
};

use common::{
    Activation,
    data::{DataType, Features},
    Data, Input, HIDDEN,
    util::sigmoid,
};

pub fn update_single_grad_cpu(
    pos: &Data,
    nnue: &NetworkParams,
    grad: &mut NetworkParams,
    error: &mut f32,
    blend: f32,
    scale: f32,
) {
    let bias = Accumulator::load_biases(nnue);
    let mut accs = [bias; 2];
    let mut activated = [[0.0; HIDDEN]; 2];
    let mut features = Features::default();

    let eval = nnue.forward(pos, &mut accs, &mut activated, &mut features);

    let result = pos.blended_result(blend, scale);

    let sigmoid = sigmoid(eval, 1.0);
    let err = (sigmoid - result) * sigmoid * (1. - sigmoid);
    *error += (sigmoid - result).powi(2);

    nnue.backprop(err, grad, &accs, &activated, &mut features);
}

impl NetworkParams {
    pub fn forward(
        &self,
        pos: &Data,
        accs: &mut [Accumulator; 2],
        activated: &mut [[f32; HIDDEN]; 2],
        features: &mut Features,
    ) -> f32 {
        for feat in pos.into_iter() {
            let (wfeat, bfeat) = Input::get_feature_indices(feat);

            features.push(wfeat, bfeat);
            accs[0].add_feature(wfeat, self);
            accs[1].add_feature(bfeat, self);
            if Input::FACTORISER {
                accs[0].add_feature(wfeat % Data::INPUTS, self);
                accs[1].add_feature(bfeat % Data::INPUTS, self);
            }
        }

        let mut eval = self[OUTPUT_BIAS];

        for i in 0..HIDDEN {
            activated[0][i] = Activation::activate(accs[0][i]);
            eval += activated[0][i] * self[OUTPUT_WEIGHTS + i];
        }

        for i in 0..HIDDEN {
            activated[1][i] = Activation::activate(accs[1][i]);
            eval += activated[1][i] * self[OUTPUT_WEIGHTS + HIDDEN + i];
        }

        eval
    }

    pub fn backprop(
        &self,
        err: f32,
        grad: &mut NetworkParams,
        accs: &[Accumulator; 2],
        activated: &[[f32; HIDDEN]; 2],
        features: &mut Features,
    ) {
        let mut components = [(0.0, 0.0); HIDDEN];

        for i in 0..HIDDEN {
            components[i] = (
                err * self[OUTPUT_WEIGHTS + i] * Activation::prime(accs[0][i]),
                err * self[OUTPUT_WEIGHTS + HIDDEN + i] * Activation::prime(accs[1][i]),
            );

            grad[FEATURE_BIAS + i] += components[i].0 + components[i].1;

            grad[OUTPUT_WEIGHTS + i] += err * activated[0][i];
            grad[OUTPUT_WEIGHTS + HIDDEN + i] += err * activated[1][i];
        }

        for (wfeat, bfeat) in features {
            let (widx, bidx) = (wfeat * HIDDEN, bfeat * HIDDEN);
            for i in 0..HIDDEN {
                grad[widx + i] += components[i].0;
                grad[bidx + i] += components[i].1;
            }
        }

        grad[OUTPUT_BIAS] += err;
    }
}