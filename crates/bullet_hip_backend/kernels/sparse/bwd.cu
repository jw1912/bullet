#include "../util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

template<OpType op, size_t stride = 1>
__global__ void sparseAffineBackwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const int32_t* inputs,
    const float* outputs,
    const float* errors)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const int32_t* thisInput = inputs + inputSize * blockIdx.y;
    const float* thisOutputs = outputs + stride * outputSize * blockIdx.y;
    const float* thisErrors = errors + stride * outputSize * blockIdx.y;

    const float ourError = op(thisOutputs[elem]) * thisErrors[elem];

    if (biasesGrad != nullptr)
        atomicAdd(&biasesGrad[elem], ourError);

    for (size_t i = 0; i < inputSize; i++) {
        const int32_t inp = thisInput[i];

        if (inp == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(inp) * outputSize + elem;
        atomicAdd(&weightsGrad[ourIdx], ourError);
    }
}

extern "C" void sparseAffineBackward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const int32_t* inputs,
    const float* outputs,
    const float* errors)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    sparseAffineBackwardKernel<primeInvIdentity><<<grid, threads>>>(maxInputSize, outputSize, weightsGrad, biasesGrad, inputs, outputs, errors);
}

template<OpType op>
void sparseAffineDualBackwardInternal(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const int32_t* stm,
    const int32_t* ntm,
    const float* outputs,
    const float* errors)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    sparseAffineBackwardKernel<op, 2><<<grid, threads>>>(maxInputSize, outputSize, weightsGrad, biasesGrad, stm, outputs, errors);
    sparseAffineBackwardKernel<op, 2><<<grid, threads>>>(maxInputSize, outputSize, weightsGrad, biasesGrad, ntm, outputs + outputSize, errors + outputSize);
}

extern "C" void sparseAffineDualBackward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const int32_t* stm,
    const int32_t* ntm,
    const float* outputs,
    const float* errors,
    const int32_t activation)
{
    switch (activation)
    {
        case 0:
            sparseAffineDualBackwardInternal<primeInvIdentity>(batchSize, maxInputSize, outputSize, weightsGrad, biasesGrad, stm, ntm, outputs, errors);
            break;
        case 1:
            sparseAffineDualBackwardInternal<primeInvReLU>(batchSize, maxInputSize, outputSize, weightsGrad, biasesGrad, stm, ntm, outputs, errors);
            break;
        case 2:
            sparseAffineDualBackwardInternal<primeInvCReLU>(batchSize, maxInputSize, outputSize, weightsGrad, biasesGrad, stm, ntm, outputs, errors);
            break;
        case 3:
            sparseAffineDualBackwardInternal<primeInvSCReLU>(batchSize, maxInputSize, outputSize, weightsGrad, biasesGrad, stm, ntm, outputs, errors);
            break;
        case 4:
            sparseAffineDualBackwardInternal<primeInvSqrReLU>(batchSize, maxInputSize, outputSize, weightsGrad, biasesGrad, stm, ntm, outputs, errors);
            break;
        case 5:
            sparseAffineDualBackwardInternal<primeInvSigmoid>(batchSize, maxInputSize, outputSize, weightsGrad, biasesGrad, stm, ntm, outputs, errors);
            break;
        default:
            std::abort();
    }
}
