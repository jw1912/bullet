use std::ffi::{c_float, c_void};

use crate::{
    bindings::{cudaFree, cudaError, cudaMemcpy, cudaMemcpyKind, cudaDeviceSynchronize, trainBatch},
    catch,
    util::{cuda_calloc, cuda_copy_to_gpu, cuda_malloc},
};

use common::{data::ChessBoardCUDA, HIDDEN};
use cpu::{NetworkParams, FEATURE_BIAS, OUTPUT_WEIGHTS, OUTPUT_BIAS};

/// # Safety
/// because i say so
pub unsafe fn calc_gradient(
    nnue: &NetworkParams,
    error: &mut f32,
    batch_size: usize,
    our_inputs: *const u16,
    opp_inputs: *const u16,
    results: *const c_float,
) -> Box<NetworkParams> {
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
        our_inputs,
        opp_inputs,
        results,
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

    let mut res = NetworkParams::new();
    let res_ptr = res.as_mut_ptr() as *mut c_void;

    catch!(cudaMemcpy(
        res_ptr,
        grad as *const c_void,
        NET_SIZE,
        cudaMemcpyKind::cudaMemcpyDeviceToHost,
    ), "memcpy");
    catch!(cudaDeviceSynchronize());

    catch!(cudaFree(grad.cast()), "free");
    catch!(cudaDeviceSynchronize());

    catch!(cudaFree(network.cast()), "free");
    catch!(cudaDeviceSynchronize());

    res
}