#ifndef BULLET_CUDA_UTILS
#define BULLET_CUDA_UTILS
#include "util.cu"
#endif

BULLET_KERNEL PairwiseMulKernel(
    const int output_size,
    const int batch_size,
    const float* input,
    float* output)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= output_size * batch_size)
        return;

    const int idxInBatch = tid / output_size;
    const int idxInOutput = tid % output_size;

    const float* thisInp = input + 2 * output_size * idxInBatch + idxInOutput;
    float* thisOut = output + output_size * idxInBatch + idxInOutput;

    thisOut[0] = thisInp[0] * thisInp[output_size];
}

BULLET_KERNEL PairwiseMulBackwardKernel(
    const int output_size,
    const int batch_size,
    const float* input,
    const float* output_grad,
    float* input_grad)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= output_size * batch_size)
        return;

    const int idxInBatch = tid / output_size;
    const int idxInOutput = tid % output_size;

    const float* thisOutputGrad = output_grad + output_size * idxInBatch + idxInOutput;
    
    const int inputOffset = 2 * output_size * idxInBatch + idxInOutput;
    const float* thisInput = input + inputOffset;
    float* thisInputGrad = input_grad + inputOffset;

    const float gradIn = thisOutputGrad[0];

    thisInputGrad[0] += gradIn * thisInput[output_size];
    thisInputGrad[output_size] += gradIn * thisInput[0];
}

//extern "C" void pairwiseMul(const int batch_size, const int output_size, const float* input, float* output)
//{
//    const int total_outputs = batch_size * output_size;
//    const int blocks = (total_outputs + threadsPerBlock - 1) / threadsPerBlock;
//    pairwiseMulKernel<<<blocks, threadsPerBlock>>>(output_size, batch_size, input, output);
//}
//
//extern "C" void backpropPairwiseMul(const int batch_size, const int output_size, const float* input, const float* output_grad, float* input_grad)
//{
//    const int total_outputs = batch_size * output_size;
//    const int blocks = (total_outputs + threadsPerBlock - 1) / threadsPerBlock;
//    pairwiseMulBackwardKernel<<<blocks, threadsPerBlock>>>(output_size, batch_size, input, output_grad, input_grad);
//}
