use crate::{
    data::Features,
    network::{Accumulator, Activation, NetworkParams},
    util::sigmoid,
    Data, HIDDEN,
};

pub fn gradients<Act: Activation>(
    positions: &[Data],
    nnue: &NetworkParams,
    error: &mut f32,
    blend: f32,
    skip_prop: f32,
    scale: f32,
) -> Box<NetworkParams> {
    let mut grad = NetworkParams::new();
    let mut rand = crate::rng::Rand::default();
    for pos in positions {
        if rand.rand(1.0) < skip_prop {
            continue;
        }

        update_single_grad::<Act>(pos, nnue, &mut grad, error, blend, scale);
    }
    grad
}

fn update_single_grad<Act: Activation>(
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

    let eval = nnue.forward::<Act>(pos, &mut accs, &mut activated, &mut features);

    let result = pos.blended_result(blend, scale);

    let sigmoid = sigmoid(eval, 1.0);
    let err = (sigmoid - result) * sigmoid * (1. - sigmoid);
    *error += (sigmoid - result).powi(2);

    nnue.backprop::<Act>(err, grad, &accs, &activated, &mut features);
}
