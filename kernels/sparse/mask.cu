#include "../util.cu"

__global__ void sparse_mask_kernel(
    const size_t rows,
    const size_t cols,
    const size_t max_active,
    const float* inputs,
    const int32_t* masks,
    float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= cols)
        return;

    const float* thisInput = inputs + rows * elem;
    const int32_t* thisMask = masks + max_active * elem;
    float* thisOutput = outputs + rows * elem;

    for (size_t i = 0; i < max_active; i++) {
        const int32_t inp = thisMask[i];

        if (inp == -1)
            break;

        thisOutput[inp] = thisInput[inp];
    }
}

__global__ void sparse_mask_backprop_kernel(
    const size_t rows,
    const size_t cols,
    const size_t max_active,
    const float* output_grads,
    const int32_t* masks,
    float* input_grads)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= cols)
        return;

    const float* thisOutputGrad = output_grads + rows * elem;
    const int32_t* thisMask = masks + max_active * elem;
    float* thisInputGrad = input_grads + rows * elem;

    for (size_t i = 0; i < max_active; i++) {
        const int32_t inp = thisMask[i];

        if (inp == -1)
            break;

        thisInputGrad[inp] += thisOutputGrad[inp];
    }
}


extern "C" void sparse_mask(
    const size_t rows,
    const size_t cols,
    const size_t max_active,
    const float* inputs,
    const int32_t* masks,
    float* outputs)
{
    const size_t blocks = (cols + threadsPerBlock - 1) / threadsPerBlock;
    sparse_mask_kernel<<<blocks, threadsPerBlock>>>(rows, cols, max_active, inputs, masks, outputs);
}

extern "C" void sparse_mask_backprop(
    const size_t rows,
    const size_t cols,
    const size_t max_active,
    const float* output_grads,
    const int32_t* masks,
    float* input_grads)
{
    const size_t blocks = (cols + threadsPerBlock - 1) / threadsPerBlock;
    sparse_mask_backprop_kernel<<<blocks, threadsPerBlock>>>(rows, cols, max_active, output_grads, masks, input_grads);
}
