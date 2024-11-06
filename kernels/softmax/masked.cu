#include "../util.cu"

__global__ void softmax_across_columns_masked_kernel(
    const size_t max_active,
    const size_t rows,
    const size_t cols,
    const int32_t* mask,
    const float* input,
    float* output)
{
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= cols)
        return;

    const int32_t* this_mask = mask + max_active * tid;
    const float* thisColumn = input + rows * tid;
    float* thisOutput = output + rows * tid;

    float maximum = thisColumn[this_mask[0]];

    for (size_t i = 1; i < max_active; i++) {
        const int32_t idx = this_mask[i];

        if (idx == -1)
            break;

        maximum = max(maximum, thisColumn[idx]);
    }

    float total = 0.0F;

    for (size_t i = 0; i < max_active; i++) {
        const int32_t idx = this_mask[i];

        if (idx == -1)
            break;

        const float exp = expf(thisColumn[idx] - maximum);
        thisOutput[idx] = exp;
        total += exp;
    }

    for (size_t i = 0; i < max_active; i++) {
        const int32_t idx = this_mask[i];

        if (idx == -1)
            break;

        thisOutput[idx] /= total;
    }
}

extern "C" void softmax_across_columns_masked(
    const size_t max_active,
    const size_t rows,
    const size_t cols,
    const int32_t* mask,
    const float* input,
    float* output)
{
    const size_t grid_x = (cols + threadsPerBlock - 1) / threadsPerBlock;
    cudaMemset(output, 0, sizeof(float) * rows * cols);
    softmax_across_columns_masked_kernel<<<grid_x, threadsPerBlock>>>(max_active, rows, cols, mask, input, output);
}