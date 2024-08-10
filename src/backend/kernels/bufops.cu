#include <cuda.h>
#include <cuda_runtime.h>

constexpr size_t threadsPerBlock = static_cast<size_t>(1024);

__device__ float ReLU(float in) { return in > 0.0F ? in : 0.0F; }
__device__ float CReLU(float in) { return in < 0.0F ? 0.0F : (in > 1.0F ? 1.0F : in); }
__device__ float SCReLU(float in) { return in < 0.0F ? 0.0F : (in > 1.0F ? 1.0F : (in * in)); }
__device__ float SqrReLU(float in) { return in < 0.0F ? 0.0F : (in * in); }

typedef float(*OpType)(float);

template<OpType op>
__global__ void bufferOperation(const size_t size, const float* in, float* out)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    out[i] = op(in[i]);
}

extern "C" void activateReLU(const size_t size, const float* in, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferOperation<ReLU><<<numBlocks, threadsPerBlock>>>(size, in, out);
}

extern "C" void activateCReLU(const size_t size, const float* in, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferOperation<CReLU><<<numBlocks, threadsPerBlock>>>(size, in, out);
}

extern "C" void activateSCReLU(const size_t size, const float* in, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferOperation<SCReLU><<<numBlocks, threadsPerBlock>>>(size, in, out);
}

extern "C" void activateSqrReLU(const size_t size, const float* in, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferOperation<SqrReLU><<<numBlocks, threadsPerBlock>>>(size, in, out);
}

__global__ void activateDualKernel(
    const size_t batchSize,
    const size_t tensorSize,
    const float* inp,
    float* out)
{
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= tensorSize)
        return;

    const float thisInp = inp[tensorSize * blockIdx.y + tid];
    float* thisOut = out + 2 * tensorSize * blockIdx.y + tid;

    thisOut[0] = CReLU(thisInp);
    thisOut[tensorSize] = SCReLU(thisInp);
}

extern "C" void activateDual(
    const size_t batchSize,
    const size_t tensorSize,
    const float* inp,
    float* out)
{
    const size_t grid_x = (tensorSize + threadsPerBlock - 1) / threadsPerBlock;
    const dim3 grid(grid_x, batchSize);

    activateDualKernel<<<grid, threadsPerBlock>>>(
        batchSize,
        tensorSize,
        inp,
        out
    );
}

__global__ void addToKernel(const size_t size, const float* in, float* out)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    out[i] += in[i];
}

extern "C" void addTo(const size_t size, const float* in, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    addToKernel<<<numBlocks, threadsPerBlock>>>(size, in, out);
}
