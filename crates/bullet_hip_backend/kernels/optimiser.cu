#include "util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

constexpr float Epsilon = 0.00000001F;

__global__ void AdamKernel(
    const size_t size,
    const float beta1,
    const float beta2,
    const float adj,
    const float rate,
    const bool denom,
    float* network,
    float* momentum,
    float* velocity,
    const float* gradients)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    const float grad = adj * gradients[i];
    momentum[i] = beta1 * momentum[i] + (1.0F - beta1) * grad;
    velocity[i] = beta2 * velocity[i] + (1.0F - beta2) * grad * grad;

    float val = momentum[i];
    if (denom)
        val /= sqrt(velocity[i]) + Epsilon;
    network[i] -= rate * val;
}

__global__ void ClipKernel(const size_t size, float* params, const float min_weight, const float max_weight) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    params[i] = min(max(params[i], min_weight), max_weight);
}

extern "C" void Adam(
    const size_t size,
    const float beta1,
    const float beta2,
    const float adj,
    const float rate,
    const bool denom,
    float* network,
    float* momentum,
    float* velocity,
    const float* gradients)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    AdamKernel<<<numBlocks, threadsPerBlock>>>(
        size,
        beta1,
        beta2,
        adj,
        rate,
        denom,
        network,
        momentum,
        velocity,
        gradients
    );
}

extern "C" void Clip(const size_t size, float* params, const float min_weight, const float max_weight) {
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    ClipKernel<<<numBlocks, threadsPerBlock>>>(
        size,
        params,
        min_weight,
        max_weight
    );
}