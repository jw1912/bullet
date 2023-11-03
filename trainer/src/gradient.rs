use std::thread;

use common::Data;

#[cfg(not(feature = "gpu"))]
use cpu::{NetworkParams, update_single_grad_cpu};

#[cfg(feature = "gpu")]
use common::data::gpu::chess::ChessBoardCUDA;

#[cfg(feature = "gpu")]
use cuda::{
    bindings::{cudaDeviceSynchronize, cudaError},
    calc_gradient,
    catch,
    CudaAllocations,
    util::cuda_copy_to_gpu,
};

#[cfg(not(feature = "gpu"))]
pub fn gradients_batch_cpu(
    batch: &[Data],
    nnue: &NetworkParams,
    error: &mut f32,
    scale: f32,
    blend: f32,
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
                    for pos in chunk {
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

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn gradients_batch_gpu(
    batch: &[Data],
    error: &mut f32,
    scale: f32,
    blend: f32,
    threads: usize,
    (
        our_inputs_ptr,
        opp_inputs_ptr,
        results_ptr,
        our_acc,
        opp_acc,
        outputs,
        grad,
        network,
        _, _,
    ): CudaAllocations
) {
    let batch_size = batch.len();
    let chunk_size = batch.len() / threads;

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

    unsafe {
        calc_gradient(
            error,
            batch_size,
            our_inputs_ptr,
            opp_inputs_ptr,
            results_ptr,
            our_acc,
            opp_acc,
            outputs,
            grad,
            network,
        )
    }
}
