#include "../util.cu"

// it is assumed that we will only be using this on matrixs with small number of columns
// (so the perf won't be terrible)
__global__ void softmax_across_columns_naive_kernel(const size_t rows, const size_t cols, const float* input, float* output)
{
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= cols)
        return;

    const float* thisColumn = input + rows * tid;
    float* thisOutput = output + rows * tid;

    float maximum = thisColumn[0];

    for (size_t i = 1; i < rows; i++) {
        maximum = max(maximum, thisColumn[i]);
    }

    float total = 0.0F;

    for (size_t i = 0; i < rows; i++) {
        const float exp = expf(thisColumn[i] - maximum);
        thisOutput[i] = exp;
        total += exp;
    }

    for (size_t i = 0; i < rows; i++) {
        thisOutput[i] /= total;
    }
}

extern "C" void softmax_across_columns(const size_t rows, const size_t cols, const float* input, float* output)
{
    const size_t grid_x = (cols + threadsPerBlock - 1) / threadsPerBlock;
    softmax_across_columns_naive_kernel<<<grid_x, threadsPerBlock>>>(rows, cols, input, output);
}
