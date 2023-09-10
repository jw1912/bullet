use std::{ffi::c_void, thread};

use crate::{
    cuda::{
        cuda_calloc,
        bindings::{cudaFree, cudaMemcpy, cudaMemcpyKind, cudaDeviceSynchronize}, cuda_copy_to_gpu, cuda_malloc,
    },
    data::{Features, gpu::chess::ChessBoardCUDA},
    network::{Accumulator, NetworkParams},
    util::sigmoid,
    Data, HIDDEN,
};

pub fn gradients_batch_cpu(
    batch: &[Data],
    nnue: &NetworkParams,
    error: &mut f32,
    scale: f32,
    blend: f32,
    skip_prop: f32,
    threads: usize,
) -> Box<NetworkParams> {
    let size = batch.len() / threads;
    let mut errors = vec![0.0; threads];
    let mut grad = NetworkParams::new();

    thread::scope(|s| {
        batch
            .chunks(size)
            .zip(errors.iter_mut())
            .map(|(chunk, error)| {
                s.spawn(move || {
                    let mut grad = NetworkParams::new();
                    let mut rand = crate::rng::Rand::default();
                    for pos in chunk {
                        if rand.rand(1.0) < skip_prop {
                            continue;
                        }

                        update_single_grad_cpu(pos, nnue, &mut grad, error, blend, scale);
                    }
                    grad
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|p| p.join().unwrap())
            .for_each(|part| *grad += &part);
    });
    *error += errors.iter().sum::<f32>();
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

#[allow(unused)]
pub unsafe fn gradients_batch_gpu(
    batch: &[Data],
    nnue: &NetworkParams,
    error: &mut f32,
    scale: f32,
    blend: f32,
    skip_prop: f32,
    threads: usize,
) -> Box<NetworkParams> {
    let size = batch.len() / threads;

    let inputs_ptr = cuda_malloc(batch.len() * std::mem::size_of::<ChessBoardCUDA>());
    let results_ptr = cuda_malloc(batch.len() * std::mem::size_of::<f32>());

    cudaDeviceSynchronize();

    let mut copy_count = 0;

    thread::scope(|s| {
        batch
            .chunks(size)
            .map(|chunk| {
                s.spawn(move || {
                    let num = chunk.len();
                    let mut inputs = Vec::with_capacity(num);
                    let mut results = Vec::with_capacity(num);

                    for pos in chunk {
                        ChessBoardCUDA::push(pos, &mut inputs, &mut results, blend, scale);
                    }

                    (inputs, results)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|p| p.join().unwrap())
            .for_each(|(inputs, results)| {
                let additional = inputs.len();

                cuda_copy_to_gpu(inputs_ptr.wrapping_add(copy_count), inputs.as_ptr(), additional);
                cuda_copy_to_gpu(results_ptr.wrapping_add(copy_count), results.as_ptr(), additional);

                copy_count += additional;
            });
    });

    const NET_SIZE: usize = std::mem::size_of::<NetworkParams>();
    let grad = cuda_calloc::<NET_SIZE>();

    //cudaForwardBatch(nnue, grad, );

    let mut res = NetworkParams::new();
    let res_ptr = res.as_mut_ptr() as *mut c_void;

    cudaMemcpy(
        res_ptr,
        grad as *mut c_void,
        NET_SIZE,
        cudaMemcpyKind::cudaMemcpyDeviceToHost
    );

    cudaFree(grad as *mut c_void);
    cudaFree(inputs_ptr as *mut c_void);
    cudaFree(results_ptr as *mut c_void);

    res
}
