/*
In-place operations on buffers.
*/
#include <cuda.h>
#include <cuda_runtime.h>

constexpr size_t threadsPerBlock = static_cast<size_t>(1024);

__device__ float ReLU(float in) { return in > 0.0F ? in : 0.0F; }
__device__ float CReLU(float in) { return in < 0.0F ? 0.0F : (in > 1.0F ? 1.0F : in); }
__device__ float SCReLU(float in) { return in < 0.0F ? 0.0F : (in > 1.0F ? 1.0F : (in * in)); }

typedef float(*OpType)(float);

template<OpType op>
__global__ void bufferOperation(const size_t bufferSize, float* buffer)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= bufferSize)
        return;

    buffer[i] = op(buffer[i]);
}

extern "C" void activateReLU(const size_t bufferSize, float* buffer)
{
    const size_t numBlocks = (bufferSize + threadsPerBlock - 1) / threadsPerBlock;
    bufferOperation<ReLU><<<numBlocks, threadsPerBlock>>>(bufferSize, buffer);
}

extern "C" void activateCReLU(const size_t bufferSize, float* buffer)
{
    const size_t numBlocks = (bufferSize + threadsPerBlock - 1) / threadsPerBlock;
    bufferOperation<CReLU><<<numBlocks, threadsPerBlock>>>(bufferSize, buffer);
}

extern "C" void activateSCReLU(const size_t bufferSize, float* buffer)
{
    const size_t numBlocks = (bufferSize + threadsPerBlock - 1) / threadsPerBlock;
    bufferOperation<SCReLU><<<numBlocks, threadsPerBlock>>>(bufferSize, buffer);
}
