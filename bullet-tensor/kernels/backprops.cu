/*
This file calculates in-place backprops for activation functions.
Given a batch of inputs `x[i]` and destinations `y[i]`, the
function for a given `op` calculates `y[i] *= op'(op_inv(x[i]))`.
*/
#include <cuda.h>
#include <cuda_runtime.h>

constexpr size_t threadsPerBlock = static_cast<size_t>(1024);

__device__ float primeReLU(float in) { return in > 0.0F ? 1.0F : 0.0F; }
__device__ float primeCReLU(float in) { return in > 0.0F && in < 1.0F ? 1.0F : 0.0F; }
__device__ float primeSCReLU(float in) { return in > 0.0F && in < 1.0F ? 2.0F * sqrt(in) : 0.0F; }

typedef float(*OpType)(float);

template<OpType op>
__global__ void bufferBackprop(const size_t bufferSize, float* buffer)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= bufferSize)
        return;

    buffer[i] *= op(buffer[i]);
}

extern "C" void backpropReLU(const size_t bufferSize, float* buffer)
{
    const size_t numBlocks = (bufferSize + threadsPerBlock - 1) / threadsPerBlock;
    bufferBackprop<primeReLU><<<numBlocks, threadsPerBlock>>>(bufferSize, buffer);
}

extern "C" void backpropCReLU(const size_t bufferSize, float* buffer)
{
    const size_t numBlocks = (bufferSize + threadsPerBlock - 1) / threadsPerBlock;
    bufferBackprop<primeCReLU><<<numBlocks, threadsPerBlock>>>(bufferSize, buffer);
}

extern "C" void backpropSCReLU(const size_t bufferSize, float* buffer)
{
    const size_t numBlocks = (bufferSize + threadsPerBlock - 1) / threadsPerBlock;
    bufferBackprop<primeSCReLU><<<numBlocks, threadsPerBlock>>>(bufferSize, buffer);
}
