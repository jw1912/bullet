#include "util.cu"

__global__ void gather_kernel(
    const size_t input_rows,
    const size_t output_rows,
    const size_t cols,
    const float* inputs,
    const int32_t* indices,
    float* outputs)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= cols * output_rows)
        return;

    const size_t idxInBatch = tid / output_rows;
    const size_t idxInOutput = tid % output_rows;

    const int32_t inpIdx = indices[idxInOutput];
    outputs[output_rows * idxInBatch + idxInOutput] = (inpIdx == -1) ? 0.0F : inputs[input_rows * idxInBatch + inpIdx];
}

__global__ void gather_backprop_kernel(
    const size_t input_rows,
    const size_t output_rows,
    const size_t cols,
    const float* output_grads,
    const int32_t* indices,
    float* input_grads)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= cols * output_rows)
        return;

    const size_t idxInBatch = tid / output_rows;
    const size_t idxInOutput = tid % output_rows;

    const int32_t inpIdx = indices[idxInOutput];

    if (inpIdx != -1)
        atomicAdd(&input_grads[input_rows * idxInBatch + inpIdx], output_grads[output_rows * idxInBatch + idxInOutput]);
}

extern "C" void gather(
    const size_t input_rows,
    const size_t output_rows,
    const size_t cols,
    const float* inputs,
    const int32_t* indices,
    float* outputs)
{
    const size_t blocks = (cols * output_rows + threadsPerBlock - 1) / threadsPerBlock;
    gather_kernel<<<blocks, threadsPerBlock>>>(input_rows, output_rows, cols, inputs, indices, outputs);
}

extern "C" void gather_backprop(
    const size_t input_rows,
    const size_t output_rows,
    const size_t cols,
    const float* output_grads,
    const int32_t* indices,
    float* input_grads)
{
    const size_t blocks = (cols * output_rows + threadsPerBlock - 1) / threadsPerBlock;
    gather_backprop_kernel<<<blocks, threadsPerBlock>>>(input_rows, output_rows, cols, output_grads, indices, input_grads);
}
