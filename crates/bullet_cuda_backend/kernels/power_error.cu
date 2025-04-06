#ifndef BULLET_CUDA_UTILS
#define BULLET_CUDA_UTILS
#include "util.cu"
#endif

BULLET_KERNEL PowerErrorKernel(
    const int size,
    const float* inputs,
    const float* results,
    float* output,
    const float power)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    output[i] = powf(abs(inputs[i] - results[i]), power);
}

BULLET_KERNEL PowerErrorBackwardKernel(
    const int size,
    const float* inputs,
    const float* results,
    const float* output_grad,
    float* input_grads,
    const float power)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    const float diff = inputs[i] - results[i];
    const float grad = power * powf(abs(diff), power - 1.0F) * output_grad[i];
    input_grads[i] += diff > 0.0F ? grad : -grad;
}
