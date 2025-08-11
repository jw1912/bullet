
__global__ void TransposeKernel(
    const int rows,
    const int cols,
    const float input_mul,
    const float output_mul,
    const float* input,
    float* output)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= (rows * cols))
        return;

    const int row = tid % rows;
    const int col = tid / rows;

    const int out_idx = row * cols + col;
    output[out_idx] = output_mul * output[out_idx] + input_mul * input[col * rows + row];
}

extern "C" void transpose(
    const int rows,
    const int cols,
    const float input_mul,
    const float output_mul,
    const float* input,
    float* output)
{
    const int threads = 512;
    const int size = rows * cols;
    const int blocks = (size + threads - 1) / threads;
    TransposeKernel<<<blocks, threads>>>(rows, cols, input_mul, output_mul, input, output);
}