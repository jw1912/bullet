#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

constexpr size_t threadsPerBlock = static_cast<size_t>(1024);
constexpr float B1 = 0.9F;
constexpr float B2 = 0.999F;
constexpr float B1P = 1.0F - B1;
constexpr float B2P = 1.0F - B2;
constexpr float Epsilon = 0.00000001F;
constexpr float MaxWeight = 1.98F;

__global__ void updateWeight(
    const size_t networkSize,
    const float decay,
    const float adj,
    const float rate,
    float* network,
    float* momentum,
    float* velocity,
    const float* gradients)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= networkSize)
        return;

    const float grad = adj * gradients[i];

    float param = network[i];
    param *= decay;

    momentum[i] = B1 * momentum[i] + B1P * grad;
    velocity[i] = B2 * velocity[i] + B2P * grad * grad;

    param -= rate * momentum[i] / (sqrt(velocity[i]) + Epsilon);
    param = min(max(param, -MaxWeight), MaxWeight);

    network[i] = param;
}

extern "C" void updateWeights(
    const size_t networkSize,
    const float decay,
    const float adj,
    const float rate,
    float* network,
    float* momentum,
    float* velocity,
    const float* gradients)
{
    const size_t numBlocks = (networkSize + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(updateWeight, dim3(numBlocks), dim3(threadsPerBlock), 0, 0,
        networkSize,
        decay,
        adj,
        rate,
        network,
        momentum,
        velocity,
        gradients
    );
}
