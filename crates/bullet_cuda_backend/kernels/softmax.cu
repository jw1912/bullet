#ifndef BULLET_CUDA_UTILS
#define BULLET_CUDA_UTILS
#include "util.cu"
#endif

// it is assumed that we will only be using this on matrixs with small number of columns
// (so the perf won't be terrible)
BULLET_KERNEL softmax(const int rows, const int cols, const float* input, float* output)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= cols)
        return;

    const float* thisColumn = input + rows * tid;
    float* thisOutput = output + rows * tid;

    float maximum = thisColumn[0];

    for (int i = 1; i < rows; i++) {
        maximum = max(maximum, thisColumn[i]);
    }

    float total = 0.0F;

    for (int i = 0; i < rows; i++) {
        const float exp = expf(thisColumn[i] - maximum);
        thisOutput[i] = exp;
        total += exp;
    }

    for (int i = 0; i < rows; i++) {
        thisOutput[i] /= total;
    }
}

BULLET_KERNEL cross_entropy(const int size, const float* pred, const float* target, float* out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    out[i] = (target[i] == 0.0F) ? 0.0F : -target[i] * logf(pred[i]);
}

BULLET_KERNEL backprop_softmax_cross_entropy(
    const int size,
    const float* softmaxed,
    const float* target,
    const float* out_grad,
    float* input_grad)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    input_grad[i] += (softmaxed[i] - target[i]) * out_grad[i];
}
