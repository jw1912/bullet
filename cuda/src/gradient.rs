use std::ffi::{c_float, c_void};

use crate::{
    bindings::{cudaFree, cudaError, cudaMemcpy, cudaMemcpyKind, cudaDeviceSynchronize, calcGradient, cudaMemset},
    catch,
    util::cuda_calloc,
};

use common::{data::{ChessBoardCUDA, CudaResult}, HIDDEN};
use cpu::{NetworkParams, FEATURE_BIAS, OUTPUT_WEIGHTS, OUTPUT_BIAS};

const NET_SIZE: usize = std::mem::size_of::<NetworkParams>();

/// # Safety
/// Error checked.
#[allow(clippy::too_many_arguments)]
pub unsafe fn calc_gradient(
    error: &mut f32,
    batch_size: usize,
    our_inputs: *const u16,
    opp_inputs: *const u16,
    results: *const CudaResult,
    our_acc: *mut c_float,
    opp_acc: *mut c_float,
    outputs: *mut c_float,
    grad: *mut c_float,
    network: *mut NetworkParams,
) {
    catch!(cudaMemset(grad as *mut c_void, 0, NET_SIZE), "memset");

    let feature_weights: *const f32 = network.cast();
    let feature_biases = feature_weights.wrapping_add(FEATURE_BIAS);
    let output_weights = feature_weights.wrapping_add(OUTPUT_WEIGHTS);
    let output_biases = feature_weights.wrapping_add(OUTPUT_BIAS);

    let feature_weights_grad = grad;
    let feature_biases_grad = feature_weights_grad.wrapping_add(FEATURE_BIAS);
    let output_weights_grad = feature_weights_grad.wrapping_add(OUTPUT_WEIGHTS);
    let output_biases_grad = feature_weights_grad.wrapping_add(OUTPUT_BIAS);

    let gpu_error = cuda_calloc(4);

    catch!(calcGradient(
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
        our_acc,
        opp_acc,
        outputs,
    ), "training");

    let mut batch_error = 0.0f32;

    catch!(cudaMemcpy(
        ((&mut batch_error) as *mut f32).cast(),
        gpu_error.cast(),
        std::mem::size_of::<f32>(),
        cudaMemcpyKind::cudaMemcpyDeviceToHost,
    ), "memcpy");
    catch!(cudaFree(gpu_error.cast()), "free");
    catch!(cudaDeviceSynchronize());

    *error += batch_error;
}