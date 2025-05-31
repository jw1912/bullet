#ifndef BULLET_CUDA_UTILS
#define BULLET_CUDA_UTILS
#include "util.cu"
#endif

BULLET_KERNEL select(
    const int batch_size,
    const int input_size,
    const int output_size,
    const int* buckets,
    const float* in,
    float* out)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= batch_size * output_size)
        return;

    const int idxInBatch = tid / output_size;
    const int idxInOutput = tid % output_size;

    const int thisBucket = buckets[idxInBatch];

    const float* thisInput = in + input_size * idxInBatch + output_size * thisBucket + idxInOutput;
    float* thisOutput = out + output_size * idxInBatch + idxInOutput;

    thisOutput[0] = thisInput[0];
}

BULLET_KERNEL select_backprop(
    const int batch_size,
    const int input_size,
    const int output_size,
    const int* buckets,
    const float* output_grad,
    float* input_grad)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= batch_size * output_size)
        return;

    const int idxInBatch = tid / output_size;
    const int idxInOutput = tid % output_size;

    const int thisBucket = buckets[idxInBatch];

    const float* thisOutputGrad = output_grad + output_size * idxInBatch + idxInOutput;
    float* thisInputGrad = input_grad + input_size * idxInBatch + output_size * thisBucket + idxInOutput;

    thisInputGrad[0] += thisOutputGrad[0];
}
