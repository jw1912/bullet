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

    let our_inputs_ptr = cuda_malloc(batch.len() * std::mem::size_of::<ChessBoardCUDA>());
    let opp_inputs_ptr = cuda_malloc(batch.len() * std::mem::size_of::<ChessBoardCUDA>());
    let results_ptr = cuda_malloc(batch.len() * std::mem::size_of::<f32>());

    cudaDeviceSynchronize();

    let mut copy_count = 0;

    thread::scope(|s| {
        batch
            .chunks(size)
            .map(|chunk| {
                s.spawn(move || {
                    let num = chunk.len();
                    let mut our_inputs = Vec::with_capacity(num);
                    let mut opp_inputs = Vec::with_capacity(num);
                    let mut results = Vec::with_capacity(num);

                    for pos in chunk {
                        ChessBoardCUDA::push(
                            pos,
                            &mut our_inputs,
                            &mut opp_inputs,
                            &mut results,
                            blend,
                            scale
                        );
                    }

                    (our_inputs, opp_inputs, results)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|p| p.join().unwrap())
            .for_each(|(our_inputs, opp_inputs, results)| {
                let additional = results.len();

                cuda_copy_to_gpu(our_inputs_ptr.wrapping_add(copy_count), our_inputs.as_ptr(), additional);
                cuda_copy_to_gpu(opp_inputs_ptr.wrapping_add(copy_count), opp_inputs.as_ptr(), additional);
                cuda_copy_to_gpu(results_ptr.wrapping_add(copy_count), results.as_ptr(), additional);

                copy_count += additional;
            });
    });

    const NET_SIZE: usize = std::mem::size_of::<NetworkParams>();
    let grad = cuda_calloc::<NET_SIZE>();

    //forwardBatch(nnue, inputs_ptr, error, outputs, std::mem::size_of::<ChessBoardCUDA>());
    //cudaDeviceSynchronize();

    //backpropBatch(nnue, grad, outputs, results_ptr);
    //cudaDeviceSynchronize();

    let mut res = NetworkParams::new();
    let res_ptr = res.as_mut_ptr() as *mut c_void;

    cudaMemcpy(
        res_ptr,
        grad as *mut c_void,
        NET_SIZE,
        cudaMemcpyKind::cudaMemcpyDeviceToHost
    );

    cudaDeviceSynchronize();

    cudaFree(grad.cast());
    cudaFree(our_inputs_ptr.cast());
    cudaFree(opp_inputs_ptr.cast());
    cudaFree(results_ptr.cast());

    cudaDeviceSynchronize();

    res
}
