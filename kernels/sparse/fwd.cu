#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

__global__ void sparseAffineForwardKernel(
    const size_t max_active,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const int32_t* inputs,
    float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const size_t inputIdx = max_active * blockIdx.y;
    const int32_t* thisInput = inputs + inputIdx;
    float* thisOutput = outputs + outputSize * blockIdx.y + elem;

    float ourElementVal = biases == nullptr ? 0.0F : biases[elem];

    for (size_t i = 0; i < max_active; i++) {
        const int32_t inp = thisInput[i];

        if (inp == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(inp) * outputSize + elem;
        ourElementVal += weights[ourIdx];
    }

    thisOutput[0] = ourElementVal;
}

__global__ void sparseAffineForwardAlignedKernel(
    const size_t max_active,
    const size_t output_size,
    const float* weights,
    const float* biases,
    const int32_t* inputs,
    float* outputs)
{
    extern __shared__ int32_t shared_input_indices[];
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (4 * elem >= output_size)
        return;

    if (threadIdx.x < max_active)
    {
        const size_t input_idx = max_active * blockIdx.y;
        const int32_t* this_input = inputs + input_idx;

        for (size_t i = threadIdx.x; i < max_active; i += blockDim.x)
        {
            shared_input_indices[i] = this_input[i];
        }
    }

    __syncthreads();

    float4 val = biases == nullptr ? make_float4(0.0F, 0.0F, 0.0F, 0.0F) : ((const float4 *)biases)[elem];

    for (size_t i = 0; i < max_active; i++) {
        const int32_t inp = shared_input_indices[i];

        if (inp == -1)
            break;

        const size_t our_idx = static_cast<size_t>(inp) * output_size / 4;
        const float4 a = ((const float4 *)weights)[our_idx + elem];

        val.x += a.x;
        val.y += a.y;
        val.z += a.z;
        val.w += a.w;
    }

    ((float4 *)outputs)[output_size * blockIdx.y / 4 + elem] = val;
}

extern "C" void sparseAffineForward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const int32_t* inputs,
    float* outputs)
{
    const size_t max_threads = 1024;
    const size_t alloc = maxInputSize * sizeof(int32_t);

    if ((outputSize % 4) == 0 && outputSize >= 128)
    {
        const size_t output4_size = (outputSize + 3) / 4; 
        const size_t threads = min(output4_size, max_threads);
        const size_t chunks = (output4_size + threads - 1) / threads;
        dim3 grid(chunks, batchSize);

        sparseAffineForwardAlignedKernel<<<grid, threads, alloc>>>(maxInputSize, outputSize, weights, biases, inputs, outputs);
    }
    else
    {
        const size_t threads = min(outputSize, max_threads);
        const size_t chunks = (outputSize + threads - 1) / threads;
        dim3 grid(chunks, batchSize);

        sparseAffineForwardKernel<<<grid, threads>>>(maxInputSize, outputSize, weights, biases, inputs, outputs);
    }
}
