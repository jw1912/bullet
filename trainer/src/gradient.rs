use std::thread;

use cpu::NetworkParams;
use common::{Data, rng::Rand};

#[cfg(not(feature = "cuda"))]
use cpu::update_single_grad_cpu;

#[cfg(feature = "cuda")]
use common::data::gpu::chess::ChessBoardCUDA;

#[cfg(feature = "cuda")]
use cuda::{
    bindings::{cudaDeviceSynchronize, cudaError, cudaFree},
    calc_gradient,
    catch,
    util::{cuda_copy_to_gpu, cuda_malloc},
};

#[cfg(not(feature = "cuda"))]
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
                    let mut rand = Rand::default();
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
    let batch_error = errors.iter().sum::<f32>();
    *error += batch_error;
    grad
}

#[cfg(feature = "cuda")]
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
    thread::scope(|s| {
        let mut copy_count = 0;
        batch
            .chunks(chunk_size)
            .map(|chunk| {
                s.spawn(move || {
                    let num = chunk.len();
                    let mut rand = Rand::default();
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
                let size = additional * ChessBoardCUDA::len();
                let offset = copy_count * ChessBoardCUDA::len();

                cuda_copy_to_gpu(our_inputs_ptr.wrapping_add(offset), our_inputs.as_ptr().cast(), size);
                cuda_copy_to_gpu(opp_inputs_ptr.wrapping_add(offset), opp_inputs.as_ptr().cast(), size);
                cuda_copy_to_gpu(results_ptr.wrapping_add(copy_count), results.as_ptr(), additional);

                copy_count += additional;
            });
    });

    let grad = calc_gradient(nnue, error, batch_size, our_inputs_ptr, opp_inputs_ptr, results_ptr);

    catch!(cudaFree(our_inputs_ptr.cast()), "free");
    catch!(cudaDeviceSynchronize());

    catch!(cudaFree(opp_inputs_ptr.cast()), "free");
    catch!(cudaDeviceSynchronize());

    catch!(cudaFree(results_ptr.cast()), "free");
    catch!(cudaDeviceSynchronize());

    grad
}
