#include "../util.cu"

__global__ void PairwiseMulKernel(
    const int stride,
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

    output[stride * idxInBatch + idxInOutput] = thisInp[0] * thisInp[output_size];
}

__global__ void PairwiseMulBackwardKernel(
    const int stride,
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

    const float gradIn = output_grad[stride * idxInBatch + idxInOutput];
    
    const int inputOffset = 2 * output_size * idxInBatch + idxInOutput;
    const float* thisInput = input + inputOffset;
    float* thisInputGrad = input_grad + inputOffset;

    thisInputGrad[0] += gradIn * thisInput[output_size];
    thisInputGrad[output_size] += gradIn * thisInput[0];
}

extern "C" void pairwiseMul(const size_t stride, const size_t batch_size, const size_t output_size, const float* input, float* output)
{
    const size_t total_outputs = batch_size * output_size;
    const size_t blocks = (total_outputs + threadsPerBlock - 1) / threadsPerBlock;
    PairwiseMulKernel<<<blocks, threadsPerBlock>>>(stride, output_size, batch_size, input, output);
}

extern "C" void backpropPairwiseMul(const size_t stride, const size_t batch_size, const size_t output_size, const float* input, const float* output_grad, float* input_grad)
{
    const size_t total_outputs = batch_size * output_size;
    const size_t blocks = (total_outputs + threadsPerBlock - 1) / threadsPerBlock;
    PairwiseMulBackwardKernel<<<blocks, threadsPerBlock>>>(stride, output_size, batch_size, input, output_grad, input_grad);
}