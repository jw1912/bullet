#include "util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif
#include "softmax/naive.cu"

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
