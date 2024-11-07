#include <iostream>
#include "../util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

template<OpType op>
__global__ void sparseAffineDualForwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const int32_t* stm,
    const int32_t* ntm,
    float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const size_t inputIdx = inputSize * blockIdx.y;
    const int32_t* thisStmInput = stm + inputIdx;
    const int32_t* thisNtmInput = ntm + inputIdx;
    float* thisOutput = outputs + 2 * outputSize * blockIdx.y + elem;

    float stmElementVal = biases[elem];
    float ntmElementVal = stmElementVal;

    for (size_t i = 0; i < inputSize; i++) {
        const int32_t stmInp = thisStmInput[i];
        const int32_t ntmInp = thisNtmInput[i];

        if (stmInp == -1 || ntmInp == -1)
            break;

        const size_t stmIdx = static_cast<size_t>(stmInp) * outputSize + elem;
        stmElementVal += weights[stmIdx];

        const size_t ntmIdx = static_cast<size_t>(ntmInp) * outputSize + elem;
        ntmElementVal += weights[ntmIdx];
    }

    thisOutput[0] = op(stmElementVal);
    thisOutput[outputSize] = op(ntmElementVal);
}

template<OpType op>
__global__ void sparseAffineDualForwardAlignedKernel(
    const size_t max_active,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const int32_t* stm,
    const int32_t* ntm,
    float* outputs)
{
    extern __shared__ int32_t shared_input_indices[];
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (4 * elem >= outputSize)
        return;

    if (threadIdx.x < max_active)
    {
        const size_t input_idx = max_active * blockIdx.y;
        const int32_t* this_stm = stm + input_idx;
        const int32_t* this_ntm = ntm + input_idx;

        for (size_t i = threadIdx.x; i < max_active; i += blockDim.x)
        {
            shared_input_indices[i] = this_stm[i];
            shared_input_indices[i + max_active] = this_ntm[i];
        }
    }

    __syncthreads();

    float4 stm_val = ((const float4 *)biases)[elem];
    float4 ntm_val = stm_val;

    for (size_t i = 0; i < max_active; i++) {
        const int32_t stm_inp = shared_input_indices[i];
        const int32_t ntm_inp = shared_input_indices[i + max_active];

        if (stm_inp == -1 || ntm_inp == -1)
            break;

        const size_t stm_idx = static_cast<size_t>(stm_inp) * outputSize / 4;
        const float4 stmw = ((const float4 *)weights)[stm_idx + elem];
        stm_val.x += stmw.x;
        stm_val.y += stmw.y;
        stm_val.z += stmw.z;
        stm_val.w += stmw.w;

        const size_t ntm_idx = static_cast<size_t>(ntm_inp) * outputSize / 4;
        const float4 ntmw = ((const float4 *)weights)[ntm_idx + elem];
        ntm_val.x += ntmw.x;
        ntm_val.y += ntmw.y;
        ntm_val.z += ntmw.z;
        ntm_val.w += ntmw.w;
    }

    const size_t offset = 2 * outputSize * blockIdx.y;

    ((float4 *)outputs)[offset / 4 + elem] = make_float4(
        op(stm_val.x),
        op(stm_val.y),
        op(stm_val.z),
        op(stm_val.w)
    );

    ((float4 *)outputs)[(offset + outputSize) / 4 + elem] = make_float4(
        op(ntm_val.x),
        op(ntm_val.y),
        op(ntm_val.z),
        op(ntm_val.w)
    );
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

        sparseAffineDualForwardAlignedKernel<op><<<grid, threads, alloc>>>(maxInputSize, outputSize, weights, biases, stm, ntm, outputs);
    }
    else
    {
        const size_t threads = min(outputSize, max_threads);
        const size_t chunks = (outputSize + threads - 1) / threads;
        dim3 grid(chunks, batchSize);

        sparseAffineDualForwardKernel<op><<<grid, threads>>>(maxInputSize, outputSize, weights, biases, stm, ntm, outputs);
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
            std::cout << "Invalid activation function!" << std::endl;
            std::abort();
    }
}
