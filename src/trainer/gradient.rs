use crate::{
    data::Features,
    network::{Accumulator, NetworkParams},
    util::sigmoid,
    Data, HIDDEN,
};

pub fn gradients_cpu(
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

        update_single_grad_cpu(pos, nnue, &mut grad, error, blend, scale);
    }
    grad
}

fn update_single_grad_cpu(
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

use std::ffi::{c_float, c_void};
use crate::cuda::{
    cuda_calloc,
    bindings::{cudaFree, cudaMemcpy, cudaMemcpyKind},
};

#[allow(unused)]
pub unsafe fn gradients_gpu(
    positions: &[Data],
    nnue: *mut c_float,
    error: &mut f32,
    blend: f32,
    skip_prop: f32,
    scale: f32,
) -> Box<NetworkParams> {
    const NET_SIZE: usize = std::mem::size_of::<NetworkParams>();
    const ACC_SIZE: usize = std::mem::size_of::<Accumulator>();
    let grad = cuda_calloc::<NET_SIZE>();

    let accs = [
        cuda_calloc::<ACC_SIZE>(),
        cuda_calloc::<ACC_SIZE>(),
    ];

    let activated = [
        cuda_calloc::<ACC_SIZE>(),
        cuda_calloc::<ACC_SIZE>(),
    ];

    let mut rand = crate::rng::Rand::default();
    for pos in positions {
        if rand.rand(1.0) < skip_prop {
            continue;
        }

        update_single_grad_gpu(
            pos,
            nnue,
            grad,
            accs,
            activated,
            error,
            blend,
            scale
        );
    }

    let mut res = NetworkParams::new();
    let res_ptr = res.as_mut_ptr() as *mut c_void;

    cudaMemcpy(
        res_ptr,
        grad as *mut c_void,
        NET_SIZE,
        cudaMemcpyKind::cudaMemcpyDeviceToHost
    );

    cudaFree(grad as *mut c_void);

    for arr in [accs, activated] {
        for ptr in arr {
            cudaFree(ptr as *mut c_void);
        }
    }

    res
}

unsafe fn update_single_grad_gpu(
    pos: &Data,
    nnue: *mut c_float,
    grad: *mut c_float,
    accs: [*mut c_float; 2],
    acctivated: [*mut c_float; 2],
    error: &mut f32,
    blend: f32,
    scale: f32,
) {



    let mut features = Features::default();

    let eval = nnue.gpu_forward(pos, accs, activated, &mut features);

    let result = pos.blended_result(blend, scale);

    let sigmoid = sigmoid(eval, 1.0);
    let err = (sigmoid - result) * sigmoid * (1. - sigmoid);
    *error += (sigmoid - result).powi(2);

    nnue.gpu_backprop(err, grad, &accs, &activated, &mut features);


}
