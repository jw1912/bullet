/*
This file calculates in-place backprops for activation functions.
Given a batch of inputs `x[i]` and destinations `y[i]`, the
function for a given `op` calculates `y[i] = x[i] * op'(y[i])`.
*/
#include <cuda.h>
#include <cuda_runtime.h>

constexpr size_t threadsPerBlock = static_cast<size_t>(1024);

__device__ float primeReLU(float in) { return in > 0.0F ? 1.0F : 0.0F; }
__device__ float primeCReLU(float in) { return in > 0.0F && in < 1.0F ? 1.0F : 0.0F; }
__device__ float primeSCReLU(float in) { return in > 0.0F && in < 1.0F ? 2.0F * in : 0.0F; }
__device__ float primeSqrReLU(float in) { return in > 0.0F ? 2.0F * in : 0.0F; }

typedef float(*OpType)(float);

template<OpType op>
__global__ void bufferBackprop(const size_t size, const float* in, float* out)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    const float thisIn = in[i];
    const float thisOut = out[i];

    out[i] = thisIn * op(thisOut);
}

extern "C" void backpropReLU(const size_t size, const float* in, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferBackprop<primeReLU><<<numBlocks, threadsPerBlock>>>(size, in, out);
}

extern "C" void backpropCReLU(const size_t size, const float* in, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferBackprop<primeCReLU><<<numBlocks, threadsPerBlock>>>(size, in, out);
}

extern "C" void backpropSCReLU(const size_t size, const float* in, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferBackprop<primeSCReLU><<<numBlocks, threadsPerBlock>>>(size, in, out);
}

extern "C" void backpropSqrReLU(const size_t size, const float* in, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferBackprop<primeSqrReLU><<<numBlocks, threadsPerBlock>>>(size, in, out);
}
