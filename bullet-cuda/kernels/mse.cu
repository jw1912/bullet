/*
Computes MSE(sigmoid(outputs), results).
*/
#include <cuda.h>
#include <cuda_runtime.h>

constexpr size_t threadsPerBlock = static_cast<size_t>(1024);

__global__ void sigmoidMSEKernel(
    const size_t bufferSize,
    float* outputs,
    const float* results,
    float* error)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= bufferSize)
        return;

    const float sigmoid = 1.0F / (1.0F + expf(-outputs[i]));
    const float diff = sigmoid - results[i];
    outputs[i] = diff * sigmoid * (1.0F - sigmoid);
    atomicAdd(error, diff * diff);
}

extern "C" void sigmoidMSE(
    const size_t bufferSize,
    float* outputs,
    const float* results,
    float* error)
{
    const size_t numBlocks = (bufferSize + threadsPerBlock - 1) / threadsPerBlock;
    sigmoidMSEKernel<<<numBlocks, threadsPerBlock>>>(bufferSize, outputs, results, error);
}
