/*
Adapted from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf.
The idea is to do a reduction on an N-element tensor by running an
(lowPow2 / 128) x N grid of blocks, each containing 128 threads.
*/
#include <cuda.h>
#include <cuda_runtime.h>

constexpr size_t threads = 1024;

__device__ void warpReduce(volatile float* sdata, const size_t tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid +  8];
    sdata[tid] += sdata[tid +  4];
    sdata[tid] += sdata[tid +  2];
    sdata[tid] += sdata[tid +  1];
}

__global__ void reduceAddKernel(
    const size_t batchSize,
    const size_t stride,
    const float* inp,
    float* out)
{
    __shared__ float sdata[threads];

    const size_t offset = blockIdx.y;
    const size_t tid = threadIdx.x;
    const size_t myId = 2 * blockDim.x * blockIdx.x + tid;

    sdata[tid] = myId < batchSize ? inp[offset + stride * myId] : 0;

    const size_t myId2 = myId + blockDim.x;
    if (myId2 < batchSize)
        sdata[tid] += inp[offset + stride * myId2];

    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32)
        warpReduce(sdata, tid);

    if (tid == 0)
        atomicAdd(&out[offset], sdata[0]);
}

extern "C" void reduceAdd(
    const size_t batchSize,
    const size_t tensorSize,
    const float* inp,
    float* out)
{
    const size_t dual = 2 * threads;
    const size_t grid_x = (batchSize + dual - 1) / dual;
    const dim3 grid(grid_x, tensorSize);

    reduceAddKernel<<<grid, threads>>>(
        batchSize,
        tensorSize,
        inp,
        out
    );
}
