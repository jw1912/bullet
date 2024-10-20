#include "util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

// it is assumed that we will only be using this on matrixs with small number of columns
// (so the perf won't be terrible)
__global__ void softmaxAcrossColumnsKernel(const size_t rows, const size_t cols, const float* input, float* output)
{
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= cols)
        return;

    const float* thisColumn = input + rows * tid;
    float* thisOutput = output + rows * tid;

    float maximum = -100000.0F;

    for (size_t i = 0; i < rows; i++) {
        maximum = max(maximum, thisColumn[i]);
    }

    float total = 0.0F;

    for (size_t i = 0; i < rows; i++) {
        const float exp = expf(thisColumn[i] - maximum);
        thisOutput[i] = exp;
        total += exp;
    }

    for (size_t i = 0; i < rows; i++) {
        thisOutput[i] /= total;
    }
}

__global__ void crossEntropyKernel(const size_t size, const float* pred, const float* target, float* out)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    out[i] = (target[i] == 0.0F) ? 0.0F : -target[i] * logf(pred[i]);
}

__global__ void backpropSoftmaxCrossEntropyKernel(
    const size_t size,
    const float* softmaxed,
    const float* target,
    const float* out_grad,
    float* input_grad)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    input_grad[i] += (softmaxed[i] - target[i]) * out_grad[0];
}

extern "C" void softmax_across_columns(const size_t rows, const size_t cols, const float* input, float* output)
{
    const size_t grid_x = (cols + threadsPerBlock - 1) / threadsPerBlock;
    softmaxAcrossColumnsKernel<<<grid_x, threadsPerBlock>>>(rows, cols, input, output);
}

extern "C" void crossentropy(const size_t size, const float* pred, const float* target, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    crossEntropyKernel<<<numBlocks, threadsPerBlock>>>(size, pred, target, out);
}

extern "C" void backprop_softmax_cross_entropy(
    const size_t size,
    const float* softmaxed,
    const float* target,
    const float* out_grad,
    float* input_grad)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    backpropSoftmaxCrossEntropyKernel<<<numBlocks, threadsPerBlock>>>(size, softmaxed, target, out_grad, input_grad);
}
