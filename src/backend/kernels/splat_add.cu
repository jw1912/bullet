#include <cuda.h>
#include <cuda_runtime.h>

constexpr size_t threads = 1024;

__global__ void splatAddKernel(
    const size_t batchSize,
    const size_t stride,
    const float* inp_a,
    const float* inp_b,
    float* out)
{
    const size_t offset = blockIdx.y;
    const size_t tid = threadIdx.x;
    const size_t myId = blockDim.x * blockIdx.x + tid;

    if (myId >= batchSize)
        return;

    const size_t idx = offset + stride * myId;

    out[idx] = inp_a[offset] + inp_b[idx];
}

extern "C" void splatAdd(
    const size_t batchSize,
    const size_t tensorSize,
    const float* inp_a,
    const float* inp_b,
    float* out)
{
    const size_t grid_x = (batchSize + threads - 1) / threads;
    const dim3 grid(grid_x, tensorSize);

    splatAddKernel<<<grid, threads>>>(
        batchSize,
        tensorSize,
        inp_a,
        inp_b,
        out
    );
}
