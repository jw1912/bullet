#include "util.cu"

__global__ void gather_kernel(
    const size_t input_rows,
    const size_t output_rows,
    const size_t cols,
    const float* inputs,
    const int32_t* indices,
    float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= output_rows)
        return;

    const size_t tid = blockIdx.y;
    const size_t outIdx = output_rows * tid + elem;
    const int32_t inpIdx = indices[outIdx];
    outputs[outIdx] = (inpIdx == -1) ? 0.0F : inputs[input_rows * tid + inpIdx];
}

__global__ void gather_backprop_kernel(
    const size_t input_rows,
    const size_t output_rows,
    const size_t cols,
    const float* output_grads,
    const int32_t* indices,
    float* input_grads)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= output_rows)
        return;

    const size_t tid = blockIdx.y;
    const size_t outIdx = output_rows * tid + elem;
    const int32_t inpIdx = indices[outIdx];

    if (inpIdx != -1)
        atomicAdd(&input_grads[input_rows * tid + inpIdx], output_grads[outIdx]);
}

extern "C" void gather(
    const size_t input_rows,
    const size_t output_rows,
    const size_t cols,
    const float* inputs,
    const int32_t* indices,
    float* outputs)
{
    const size_t threads = min(output_rows, threadsPerBlock);
    const size_t chunks = (output_rows + threads - 1) / threads;
    dim3 grid(chunks, cols);

    gather_kernel<<<grid, threads>>>(input_rows, output_rows, cols, inputs, indices, outputs);
}

extern "C" void gather_backprop(
    const size_t input_rows,
    const size_t output_rows,
    const size_t cols,
    const float* output_grads,
    const int32_t* indices,
    float* input_grads)
{
    const size_t threads = min(output_rows, threadsPerBlock);
    const size_t chunks = (output_rows + threads - 1) / threads;
    dim3 grid(chunks, cols);

    gather_backprop_kernel<<<grid, threads>>>(input_rows, output_rows, cols, output_grads, indices, input_grads);
}
