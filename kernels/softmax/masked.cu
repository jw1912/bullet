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
    float* thisOutput = output + max_active * tid;

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
        thisOutput[i] = exp;
        total += exp;
    }

    for (size_t i = 0; i < max_active; i++) {
        if (this_mask[i] == -1)
            break;

        thisOutput[i] /= total;
    }
}

__global__ void cross_entropy_masked_kernel(
    const size_t max_active,
    const size_t rows,
    const size_t cols,
    const int32_t* mask,
    const float* pred,
    const float* target,
    float* out,
    float* error)
{
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= cols)
        return;

    const int32_t* this_mask = mask + max_active * tid;
    const float* this_pred = pred + max_active * tid;
    const float* this_target = target + max_active * tid;
    float* this_out = out + max_active * tid;

    for (size_t i = 0; i < max_active; i++) {
        if (this_mask[i] == -1)
            break;

        const float err = (this_target[i] == 0.0F) ? 0.0F : -this_target[i] * logf(this_pred[i]);
        this_out[i] = err;
        atomicAdd(error, err);
    }
}

__global__ void backprop_softmax_cross_entropy_masked_kernel(
    const size_t max_active,
    const size_t rows,
    const size_t cols,
    const int32_t* mask,
    const float* softmaxed,
    const float* target,
    const float* out_grad,
    float* input_grad)
{
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= cols)
        return;

    const int32_t* this_mask = mask + max_active * tid;
    const float* this_smax = softmaxed + max_active * tid;
    const float* this_target = target + max_active * tid;
    float* this_grad = input_grad + rows * tid;

    for (size_t i = 0; i < max_active; i++) {
        const int32_t idx = this_mask[i];

        if (idx == -1)
            break;

        this_grad[idx] += (this_smax[i] - this_target[i]) * out_grad[0];
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
    softmax_across_columns_masked_kernel<<<grid_x, threadsPerBlock>>>(max_active, rows, cols, mask, input, output);
}

extern "C" void crossentropy_masked(
    const size_t max_active,
    const size_t rows,
    const size_t cols,
    const int32_t* mask,
    const float* pred,
    const float* target,
    float* out,
    float* error)
{
    const size_t grid_x = (cols + threadsPerBlock - 1) / threadsPerBlock;
    cross_entropy_masked_kernel<<<grid_x, threadsPerBlock>>>(max_active, rows, cols, mask, pred, target, out, error);
}

extern "C" void backprop_softmax_cross_entropy_masked(
    const size_t max_active,
    const size_t rows,
    const size_t cols,
    const int32_t* mask,
    const float* softmaxed,
    const float* target,
    const float* out_grad,
    float* input_grad)
{
    const size_t grid_x = (cols + threadsPerBlock - 1) / threadsPerBlock;
    backprop_softmax_cross_entropy_masked_kernel<<<grid_x, threadsPerBlock>>>(max_active, rows, cols, mask, softmaxed, target, out_grad, input_grad);
}