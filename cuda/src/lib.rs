pub mod bindings;
mod gradient;
mod optimiser;
pub mod util;

use bindings::{cudaFree, cudaError};
pub use gradient::calc_gradient;
pub use optimiser::update_weights;
use util::{cuda_calloc, cuda_malloc, cuda_copy_from_gpu};

use common::{data::{BoardCUDA, CudaResult}, HIDDEN};
use cpu::NetworkParams;

const NET_SIZE: usize = std::mem::size_of::<NetworkParams>();

pub type CudaAllocations = (
    *mut u16, *mut u16,
    *mut CudaResult,
    *mut f32, *mut f32,
    *mut f32,
    *mut f32,
    *mut NetworkParams,
    *mut f32, *mut f32,
);

pub fn preallocate(batch_size: usize) -> CudaAllocations {
    const F32: usize = std::mem::size_of::<f32>();
    const RESULT: usize = std::mem::size_of::<CudaResult>();
    const INPUT_SIZE: usize = std::mem::size_of::<BoardCUDA>();

    let our_inputs = cuda_malloc(batch_size * INPUT_SIZE);
    let opp_inputs = cuda_malloc(batch_size * INPUT_SIZE);
    let results = cuda_malloc(batch_size * RESULT);
    let our_acc = cuda_malloc(batch_size * HIDDEN * F32);
    let opp_acc = cuda_malloc(batch_size * HIDDEN * F32);
    let outputs = cuda_malloc(batch_size * F32);
    let grad = cuda_malloc(NET_SIZE);
    let network = cuda_malloc(NET_SIZE);
    let momentum = cuda_calloc(NET_SIZE);
    let velocity = cuda_calloc(NET_SIZE);

    (our_inputs, opp_inputs, results, our_acc, opp_acc, outputs, grad, network, momentum, velocity)
}

pub fn free_preallocations(ptrs: CudaAllocations) {
    catch!(cudaFree(ptrs.0.cast()), "free");
    catch!(cudaFree(ptrs.1.cast()), "free");
    catch!(cudaFree(ptrs.2.cast()), "free");
    catch!(cudaFree(ptrs.3.cast()), "free");
    catch!(cudaFree(ptrs.4.cast()), "free");
    catch!(cudaFree(ptrs.5.cast()), "free");
    catch!(cudaFree(ptrs.6.cast()), "free");
    catch!(cudaFree(ptrs.7.cast()), "free");
    catch!(cudaFree(ptrs.8.cast()), "free");
    catch!(cudaFree(ptrs.9.cast()), "free");
}

pub fn copy_weights_from_gpu(
    cpu_network: &mut NetworkParams,
    cpu_momentum: &mut NetworkParams,
    cpu_velocity: &mut NetworkParams,
    (_, _, _, _, _, _, _, network, momentum, velocity): CudaAllocations,
) {
    cuda_copy_from_gpu(cpu_network, network, 1);
    cuda_copy_from_gpu(cpu_momentum, momentum.cast(), 1);
    cuda_copy_from_gpu(cpu_velocity, velocity.cast(), 1);
}