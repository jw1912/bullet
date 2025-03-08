#include "../util.cu"

__global__ void pairwiseMulKernel(
    const size_t output_size,
    const size_t batch_size,
    const float* input,
    float* output)
{
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= output_size * batch_size)
        return;

    const size_t idxInBatch = tid / output_size;
    const size_t idxInOutput = tid % output_size;

    const float* thisInp = input + 2 * output_size * idxInBatch + idxInOutput;
    float* thisOut = output + output_size * idxInBatch + idxInOutput;

    thisOut[0] = thisInp[0] * thisInp[output_size];
}

__global__ void pairwiseMulBackwardKernel(
    const size_t output_size,
    const size_t batch_size,
    const float* input,
    const float* output_grad,
    float* input_grad)
{
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= output_size * batch_size)
        return;

    const size_t idxInBatch = tid / output_size;
    const size_t idxInOutput = tid % output_size;

    const float* thisOutputGrad = output_grad + output_size * idxInBatch + idxInOutput;
    
    const size_t inputOffset = 2 * output_size * idxInBatch + idxInOutput;
    const float* thisInput = input + inputOffset;
    float* thisInputGrad = input_grad + inputOffset;

    const float gradIn = thisOutputGrad[0];

    thisInputGrad[0] += gradIn * thisInput[output_size];
    thisInputGrad[output_size] += gradIn * thisInput[0];
}

extern "C" void pairwiseMul(const size_t batch_size, const size_t output_size, const float* input, float* output)
{
    const size_t total_outputs = batch_size * output_size;
    const size_t blocks = (total_outputs + threadsPerBlock - 1) / threadsPerBlock;
    pairwiseMulKernel<<<blocks, threadsPerBlock>>>(output_size, batch_size, input, output);
}

extern "C" void backpropPairwiseMul(const size_t batch_size, const size_t output_size, const float* input, const float* output_grad, float* input_grad)
{
    const size_t total_outputs = batch_size * output_size;
    const size_t blocks = (total_outputs + threadsPerBlock - 1) / threadsPerBlock;
    pairwiseMulBackwardKernel<<<blocks, threadsPerBlock>>>(output_size, batch_size, input, output_grad, input_grad);
}