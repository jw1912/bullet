/*
Updating network weights given a gradient.
*/

#include <cuda.h>
#include <cuda_runtime.h>

#include "util.h"

constexpr float B1 = 0.9;
constexpr float B2 = 0.999;

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

    if (i > networkSize)
        return;

    const float grad = adj * gradients[i];

    float param = network[i];
    param *= decay;

    momentum[i] = B1 * momentum[i] + (1.0 - B1) * grad;
    velocity[i] = B2 * velocity[i] + (1.0 - B2) * grad * grad;

    param -= rate * momentum[i] / (sqrt(velocity[i] + 0.00000001));
    param = min(max(param, -1.98), 1.98);

    network[i] = param;
}

extern "C" cudaError updateWeights(
    const size_t networkSize,
    const float decay,
    const float adj,
    const float rate,
    float* network,
    float* momentum,
    float* velocity,
    const float* gradients)
{
    const size_t blockSize = calcBlocks(networkSize, 1024);
    updateWeight<<<blockSize, 1024>>>(
        networkSize,
        decay,
        adj,
        rate,
        network,
        momentum,
        velocity,
        gradients
    );

    cudaDeviceSynchronize();

    return cudaGetLastError();
}