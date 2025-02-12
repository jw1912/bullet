#include "../util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

template<OpType op, size_t stride = 1>
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
    float* thisOutput = outputs + stride * outputSize * blockIdx.y + elem;

    float ourElementVal = biases == nullptr ? 0.0F : biases[elem];

    for (size_t i = 0; i < max_active; i++) {
        const int32_t inp = thisInput[i];

        if (inp == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(inp) * outputSize + elem;
        ourElementVal += weights[ourIdx];
    }

    thisOutput[0] = op(ourElementVal);
}

template<OpType op, size_t stride = 1>
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

    val.x = op(val.x);
    val.y = op(val.y);
    val.z = op(val.z);
    val.w = op(val.w);

    ((float4 *)outputs)[stride * output_size * blockIdx.y / 4 + elem] = val;
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

        sparseAffineForwardAlignedKernel<Identity><<<grid, threads, alloc>>>(maxInputSize, outputSize, weights, biases, inputs, outputs);
    }
    else
    {
        const size_t threads = min(outputSize, max_threads);
        const size_t chunks = (outputSize + threads - 1) / threads;
        dim3 grid(chunks, batchSize);

        sparseAffineForwardKernel<Identity><<<grid, threads>>>(maxInputSize, outputSize, weights, biases, inputs, outputs);
    }
}

template<OpType op>
void sparseAffineDualForwardInternal(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const int32_t* stm,
    const int32_t* ntm,
    float* outputs)
{
    const size_t max_threads = 1024;
    const size_t alloc = 2 * maxInputSize * sizeof(int32_t);
    
    if ((outputSize % 4) == 0 && outputSize >= 128)
    {
        const size_t output4_size = (outputSize + 3) / 4; 
        const size_t threads = min(output4_size, max_threads);
        const size_t chunks = (output4_size + threads - 1) / threads;
        dim3 grid(chunks, batchSize);

        sparseAffineForwardAlignedKernel<op, 2><<<grid, threads, alloc>>>(maxInputSize, outputSize, weights, biases, stm, outputs);
        sparseAffineForwardAlignedKernel<op, 2><<<grid, threads, alloc>>>(maxInputSize, outputSize, weights, biases, ntm, outputs + outputSize);
    }
    else
    {
        const size_t threads = min(outputSize, max_threads);
        const size_t chunks = (outputSize + threads - 1) / threads;
        dim3 grid(chunks, batchSize);

        sparseAffineForwardKernel<op, 2><<<grid, threads>>>(maxInputSize, outputSize, weights, biases, stm, outputs);
        sparseAffineForwardKernel<op, 2><<<grid, threads>>>(maxInputSize, outputSize, weights, biases, ntm, outputs + outputSize);
    }
}

extern "C" void sparseAffineDualForward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const int32_t* stm,
    const int32_t* ntm,
    float* outputs,
    const int32_t activation)
{
    switch (activation)
    {
        case 0:
            sparseAffineDualForwardInternal<Identity>(batchSize, maxInputSize, outputSize, weights, biases, stm, ntm, outputs);
            break;
        case 1:
            sparseAffineDualForwardInternal<ReLU>(batchSize, maxInputSize, outputSize, weights, biases, stm, ntm, outputs);
            break;
        case 2:
            sparseAffineDualForwardInternal<CReLU>(batchSize, maxInputSize, outputSize, weights, biases, stm, ntm, outputs);
            break;
        case 3:
            sparseAffineDualForwardInternal<SCReLU>(batchSize, maxInputSize, outputSize, weights, biases, stm, ntm, outputs);
            break;
        case 4:
            sparseAffineDualForwardInternal<SqrReLU>(batchSize, maxInputSize, outputSize, weights, biases, stm, ntm, outputs);
            break;
        case 5:
            sparseAffineDualForwardInternal<sigmoid>(batchSize, maxInputSize, outputSize, weights, biases, stm, ntm, outputs);
            break;
        default:
            std::abort();
    }
}
