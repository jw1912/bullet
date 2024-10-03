#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

template<bool bias>
__global__ void sparseAffineForwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const int32_t* inputs,
    float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const size_t inputIdx = inputSize * blockIdx.y;
    const int32_t* thisInput = inputs + inputSize * blockIdx.y;
    float* thisOutput = outputs + outputSize * blockIdx.y + elem;

    float ourElementVal;
    if constexpr (bias)
        ourElementVal = biases[elem];
    else
        ourElementVal = 0.0F;

    for (size_t i = 0; i < inputSize; i++) {
        const int32_t inp = thisInput[i];

        if (inp == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(inp) * outputSize + elem;
        ourElementVal += weights[ourIdx];
    }

    thisOutput[0] = ourElementVal;
}

template<bool bias>
__global__ void sparseAffineBackwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const int32_t* inputs,
    const float* errors)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const int32_t* thisInput = inputs + inputSize * blockIdx.y;
    const float* thisErrors = errors + outputSize * blockIdx.y;

    float ourError = thisErrors[elem];

    if constexpr (bias)
        atomicAdd(&biasesGrad[elem], ourError);

    for (size_t i = 0; i < inputSize; i++) {
        const int32_t inp = thisInput[i];

        if (inp == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(inp) * outputSize + elem;
        atomicAdd(&weightsGrad[ourIdx], ourError);
    }
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
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    if (biases == nullptr)
        sparseAffineForwardKernel<false><<<grid, threads>>>(
            maxInputSize,
            outputSize,
            weights,
            biases,
            inputs,
            outputs
        );
    else
        sparseAffineForwardKernel<true><<<grid, threads>>>(
            maxInputSize,
            outputSize,
            weights,
            biases,
            inputs,
            outputs
        );
}

extern "C" void sparseAffineBackward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const int32_t* inputs,
    const float* errors)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    if (biasesGrad == nullptr)
        sparseAffineBackwardKernel<false><<<grid, threads>>>(
            maxInputSize,
            outputSize,
            weightsGrad,
            biasesGrad,
            inputs,
            errors
        );
    else
        sparseAffineBackwardKernel<true><<<grid, threads>>>(
            maxInputSize,
            outputSize,
            weightsGrad,
            biasesGrad,
            inputs,
            errors
        );
}
