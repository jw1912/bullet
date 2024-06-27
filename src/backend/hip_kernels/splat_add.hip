#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

constexpr size_t threads = 1024;

__global__ void splatAddKernel(
    const size_t batchSize,
    const size_t stride,
    const float* inp,
    float* out)
{
    const size_t offset = blockIdx.y;
    const size_t tid = threadIdx.x;
    const size_t myId = blockDim.x * blockIdx.x + tid;

    if (myId >= batchSize)
        return;

    out[offset + stride * myId] += inp[offset];
}

extern "C" void splatAdd(
    const size_t batchSize,
    const size_t tensorSize,
    const float* inp,
    float* out)
{
    const size_t grid_x = (batchSize + threads - 1) / threads;
    const dim3 grid(grid_x, tensorSize);

    hipLaunchKernelGGL(splatAddKernel, dim3(grid), dim3(threads), 0, 0,
        batchSize,
        tensorSize,
        inp,
        out
    );
}
