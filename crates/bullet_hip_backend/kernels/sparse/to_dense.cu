__global__ void sparseToDenseKernel(const size_t rows, const size_t cols, const size_t max_active, const int32_t* inputs, float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= cols)
        return;

    const int32_t* thisInput = inputs + max_active * elem;
    float* thisOutput = outputs + rows * elem;

    for (size_t i = 0; i < max_active; i++) {
        const int32_t inp = thisInput[i];

        if (inp == -1)
            break;

        thisOutput[inp] = 1.0F;
    }
}

extern "C" void sparse_to_dense(const size_t rows, const size_t cols, const size_t max_active, const int32_t* inputs, float* outputs)
{
    const size_t max_threads = 1024;
    const size_t threads = min(cols, max_threads);
    const size_t blocks = (cols + threads - 1) / threads;
    
    sparseToDenseKernel<<<blocks, threads>>>(rows, cols, max_active, inputs, outputs);
}
