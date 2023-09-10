use std::ffi::c_float;

use super::{
    Accumulator,
    NetworkParams,
    FEATURE_BIAS,
    OUTPUT_BIAS,
    OUTPUT_WEIGHTS,
    InputType
};

use crate::{
    Activation,
    data::{DataType, Features},
    Data, Input, HIDDEN,
};

pub fn gpu_forward(
    pos: &Data,
    nnue: *mut c_float,
    accs: [*mut f32; 2],
    activated: [*mut f32; 2],
    features: &mut Features,
) -> f32 {
    for feat in pos.into_iter() {
        let (wfeat, bfeat) = Input::get_feature_indices(feat);
        features.push(wfeat, bfeat);
        //accs[0].add_feature(wfeat, self);
        //accs[1].add_feature(bfeat, self);
        //if Input::FACTORISER {
        //    accs[0].add_feature(wfeat % Data::INPUTS, self);
        //    accs[1].add_feature(bfeat % Data::INPUTS, self);
        //}
    }

    //let mut eval = self[OUTPUT_BIAS];

    //for i in 0..HIDDEN {
    //    activated[0][i] = Activation::activate(accs[0][i]);
    //    eval += activated[0][i] * self[OUTPUT_WEIGHTS + i];
    //}

    //for i in 0..HIDDEN {
    //    activated[1][i] = Activation::activate(accs[1][i]);
    //    eval += activated[1][i] * self[OUTPUT_WEIGHTS + HIDDEN + i];
    //}

    //eval
    0.0
}

pub fn gpu_backprop(
    err: f32,
    nnue: *mut c_float,
    grad: *mut c_float,
    accs: [*mut f32; 2],
    activated: [*mut f32; 2],
    features: &mut Features,
) {
    /*
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
    */
}
