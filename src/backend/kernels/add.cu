#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

constexpr size_t threads = 1024;

__global__ void splatAddKernel(
    const size_t cols,
    const size_t stride,
    const float* inp_a,
    const float* inp_b,
    float* out)
{
    const size_t offset = blockIdx.y;
    const size_t tid = threadIdx.x;
    const size_t myId = blockDim.x * blockIdx.x + tid;

    if (myId >= cols)
        return;

    const size_t idx = offset + stride * myId;

    out[idx] = inp_a[offset] + inp_b[idx];
}

extern "C" void splat_add(
    const size_t cols,
    const size_t rows,
    const float* inp_a,
    const float* inp_b,
    float* out)
{
    const size_t grid_x = (cols + threads - 1) / threads;
    const dim3 grid(grid_x, rows);

    splatAddKernel<<<grid, threads>>>(
        cols,
        rows,
        inp_a,
        inp_b,
        out
    );
}
