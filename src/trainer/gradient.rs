use std::{ffi::c_void, thread};

use crate::{
    catch,
    cuda::{
        cuda_calloc, cuda_copy_to_gpu, cuda_malloc,
        bindings::{cudaFree, cudaError, cudaMemcpy, cudaMemcpyKind, cudaDeviceSynchronize, trainBatch},
    },
    data::{Features, gpu::chess::ChessBoardCUDA},
    network::{Accumulator, NetworkParams, FEATURE_BIAS, OUTPUT_BIAS, OUTPUT_WEIGHTS},
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

pub unsafe fn gradients_batch_gpu(
    batch: &[Data],
    nnue: &NetworkParams,
    error: &mut f32,
    scale: f32,
    blend: f32,
    skip_prop: f32,
    threads: usize,
) -> Box<NetworkParams> {
    let batch_size = batch.len();
    let chunk_size = batch.len() / threads;

    const INPUT_SIZE: usize = std::mem::size_of::<ChessBoardCUDA>();

    let our_inputs_ptr = cuda_malloc::<u16>(batch_size * INPUT_SIZE);
    let opp_inputs_ptr = cuda_malloc::<u16>(batch_size * INPUT_SIZE);
    let results_ptr = cuda_malloc::<f32>(batch_size * std::mem::size_of::<f32>());

    catch!(cudaDeviceSynchronize());

    let mut copy_count = 0;

    thread::scope(|s| {
        batch
            .chunks(chunk_size)
            .map(|chunk| {
                s.spawn(move || {
                    let num = chunk.len();
                    let mut rand = crate::rng::Rand::default();
                    let mut our_inputs = Vec::with_capacity(num);
                    let mut opp_inputs = Vec::with_capacity(num);
                    let mut results = Vec::with_capacity(num);

                    for pos in chunk {
                        if rand.rand(1.0) < skip_prop {
                            continue;
                        }

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

                cuda_copy_to_gpu(our_inputs_ptr.wrapping_add(copy_count), our_inputs.as_ptr().cast(), additional);
                cuda_copy_to_gpu(opp_inputs_ptr.wrapping_add(copy_count), opp_inputs.as_ptr().cast(), additional);
                cuda_copy_to_gpu(results_ptr.wrapping_add(copy_count), results.as_ptr(), additional);

                copy_count += additional;
            });
    });

    const NET_SIZE: usize = std::mem::size_of::<NetworkParams>();
    let grad = cuda_calloc::<NET_SIZE>();

    let network = cuda_malloc(NET_SIZE);
    cuda_copy_to_gpu(network, nnue as *const NetworkParams, 1);

    let feature_weights: *const f32 = (network as *const NetworkParams).cast();
    let feature_biases = feature_weights.wrapping_add(FEATURE_BIAS);
    let output_weights = feature_weights.wrapping_add(OUTPUT_WEIGHTS);
    let output_biases = feature_weights.wrapping_add(OUTPUT_BIAS);

    let feature_weights_grad: *mut f32 = (grad as *mut NetworkParams).cast();
    let feature_biases_grad = feature_weights_grad.wrapping_add(FEATURE_BIAS);
    let output_weights_grad = feature_weights_grad.wrapping_add(OUTPUT_WEIGHTS);
    let output_biases_grad = feature_weights_grad.wrapping_add(OUTPUT_BIAS);

    let gpu_error = cuda_calloc::<4>();

    catch!(trainBatch(
        batch_size,
        HIDDEN,
        ChessBoardCUDA::len(),
        feature_weights,
        feature_biases,
        output_weights,
        output_biases,
        our_inputs_ptr,
        opp_inputs_ptr,
        results_ptr,
        feature_weights_grad,
        feature_biases_grad,
        output_weights_grad,
        output_biases_grad,
        gpu_error,
    ), "training");

    let mut batch_error = 0.0f32;

    catch!(cudaMemcpy(
        ((&mut batch_error) as *mut f32).cast(),
        gpu_error.cast(),
        std::mem::size_of::<f32>(),
        cudaMemcpyKind::cudaMemcpyDeviceToHost,
    ), "memcpy");
    catch!(cudaDeviceSynchronize());

    *error += batch_error;

    println!("{batch_error}");

    let mut res = NetworkParams::new();
    let res_ptr = res.as_mut_ptr() as *mut c_void;

    catch!(cudaMemcpy(
        res_ptr,
        grad as *mut c_void,
        NET_SIZE,
        cudaMemcpyKind::cudaMemcpyDeviceToHost,
    ), "memcpy");
    catch!(cudaDeviceSynchronize());

    catch!(cudaFree(grad.cast()), "free");
    catch!(cudaDeviceSynchronize());

    catch!(cudaFree(our_inputs_ptr.cast()), "free");
    catch!(cudaDeviceSynchronize());

    catch!(cudaFree(opp_inputs_ptr.cast()), "free");
    catch!(cudaDeviceSynchronize());

    catch!(cudaFree(results_ptr.cast()), "free");
    catch!(cudaDeviceSynchronize());

    catch!(cudaFree(network.cast()), "free");
    catch!(cudaDeviceSynchronize());

    res
}
