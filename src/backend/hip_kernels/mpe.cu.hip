/*
Computes MSE(sigmoid(outputs), results).
*/
#include <hip/hip_runtime.h>


constexpr size_t threadsPerBlock = static_cast<size_t>(1024);

__global__ void sigmoidMPEKernel(
    const size_t bufferSize,
    float* outputs,
    const float* results,
    float* error,
    const float power)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= bufferSize)
        return;

    const float sigmoid = 1.0F / (1.0F + expf(-outputs[i]));
    const float diff = sigmoid - results[i];
    const float absd = abs(diff);

    outputs[i] = powf(absd, power - 1.0F) * sigmoid * (1.0F - sigmoid);
    outputs[i] = diff > 0.0F ? outputs[i] : -outputs[i];

    atomicAdd(error, powf(absd, power));
}

extern "C" void sigmoidMPE(
    const size_t bufferSize,
    float* outputs,
    const float* results,
    float* error,
    const float power)
{
    const size_t numBlocks = (bufferSize + threadsPerBlock - 1) / threadsPerBlock;
    sigmoidMPEKernel<<<numBlocks, threadsPerBlock>>>(bufferSize, outputs, results, error, power);
}
