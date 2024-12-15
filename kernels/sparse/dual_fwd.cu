#include <iostream>
#include "fwd.cu"
#include "../util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

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
        sparseAffineForwardKernel<op, 2><<<grid, threads>>>(maxInputSize, outputSize, weights, biases, stm, outputs + outputSize);
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
