use crate::{bindings::{cudaError, updateWeights}, catch, CudaAllocations};

use cpu::NETWORK_SIZE;

/// # Safety
/// Error checked.
pub unsafe fn update_weights(
    adj: f32,
    decay_rate: f32,
    rate: f32,
    (_, _, _, _, _, _, gradients, network, momentum, velocity): CudaAllocations
) {
    let decay = 1.0 - decay_rate * rate;

    catch!(updateWeights(
        NETWORK_SIZE,
        decay,
        adj,
        rate,
        network.cast(),
        momentum,
        velocity,
        gradients
    ));
}