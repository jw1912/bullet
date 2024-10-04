#include "util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

constexpr float Epsilon = 0.00000001F;

__global__ void AdamWKernel(
    const size_t size,
    const float decay,
    const float beta1,
    const float beta2,
    const float minWeight,
    const float maxWeight,
    const float adj,
    const float rate,
    float* network,
    float* momentum,
    float* velocity,
    const float* gradients)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    const float grad = adj * gradients[i];

    float param = network[i];
    param *= decay;

    momentum[i] = beta1 * momentum[i] + (1.0F - beta1) * grad;
    velocity[i] = beta2 * velocity[i] + (1.0F - beta2) * grad * grad;

    param -= rate * momentum[i] / (sqrt(velocity[i]) + Epsilon);
    param = min(max(param, minWeight), maxWeight);

    network[i] = param;
}

extern "C" void AdamW(
    const size_t size,
    const float decay,
    const float beta1,
    const float beta2,
    const float minWeight,
    const float maxWeight,
    const float adj,
    const float rate,
    float* network,
    float* momentum,
    float* velocity,
    const float* gradients)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    AdamWKernel<<<numBlocks, threadsPerBlock>>>(
        size,
        decay,
        beta1,
        beta2,
        minWeight,
        maxWeight,
        adj,
        rate,
        network,
        momentum,
        velocity,
        gradients
    );
}