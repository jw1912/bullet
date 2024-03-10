#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

struct Feat {
    uint16_t our;
    uint16_t opp;
};

__global__ void SingleSparseAffineForwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const Feat* inputs,
    float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const size_t inputIdx = inputSize * blockIdx.y;
    const Feat* thisInput = inputs + inputSize * blockIdx.y;
    float* thisOutput = outputs + outputSize * blockIdx.y + elem;

    float ourElementVal = biases[elem];

    for (size_t i = 0; i < inputSize; i++) {
        const Feat inp = thisInput[i];

        if (static_cast<size_t>(inp.our) == static_cast<size_t>(65535))
            break;

        const size_t ourIdx = static_cast<size_t>(inp.our) * outputSize + elem;
        ourElementVal += weights[ourIdx];
    }

    thisOutput[0] = ourElementVal;
}

__global__ void SingleSparseAffineBackwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const Feat* inputs,
    const float* errors)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const Feat* thisInput = inputs + inputSize * blockIdx.y;
    const float* thisErrors = errors + outputSize * blockIdx.y;

    const float ourError = thisErrors[elem];

    atomicAdd(&biasesGrad[elem], ourError);

    for (size_t i = 0; i < inputSize; i++) {
        const Feat inp = thisInput[i];

        if (static_cast<size_t>(inp.our) == static_cast<size_t>(65535))
            break;

        const size_t ourIdx = static_cast<size_t>(inp.our) * outputSize + elem;
        atomicAdd(&weightsGrad[ourIdx], ourError);
    }
}

__global__ void sparseAffineForwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const Feat* inputs,
    float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const size_t inputIdx = inputSize * blockIdx.y;
    const Feat* thisInput = inputs + inputSize * blockIdx.y;
    float* thisOutput = outputs + 2 * outputSize * blockIdx.y + elem;

    float ourElementVal = biases[elem];
    float oppElementVal = ourElementVal;

    for (size_t i = 0; i < inputSize; i++) {
        const Feat inp = thisInput[i];

        if (static_cast<size_t>(inp.our) == static_cast<size_t>(65535))
            break;

        const size_t ourIdx = static_cast<size_t>(inp.our) * outputSize + elem;
        const size_t oppIdx = static_cast<size_t>(inp.opp) * outputSize + elem;
        ourElementVal += weights[ourIdx];
        oppElementVal += weights[oppIdx];
    }

    thisOutput[         0] = ourElementVal;
    thisOutput[outputSize] = oppElementVal;
}

__global__ void sparseAffineBackwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const Feat* inputs,
    const float* errors)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const Feat* thisInput = inputs + inputSize * blockIdx.y;
    const float* thisErrors = errors + 2 * outputSize * blockIdx.y;

    const float ourError = thisErrors[elem];
    const float oppError = thisErrors[elem + outputSize];

    atomicAdd(&biasesGrad[elem], ourError + oppError);

    for (size_t i = 0; i < inputSize; i++) {
        const Feat inp = thisInput[i];

        if (static_cast<size_t>(inp.our) == static_cast<size_t>(65535))
            break;

        const size_t ourIdx = static_cast<size_t>(inp.our) * outputSize + elem;
        const size_t oppIdx = static_cast<size_t>(inp.opp) * outputSize + elem;
        atomicAdd(&weightsGrad[ourIdx], ourError);
        atomicAdd(&weightsGrad[oppIdx], oppError);
    }
}

extern "C" void singleSparseAffineForward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const Feat* inputs,
    float* outputs)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    SingleSparseAffineForwardKernel<<<grid, threads>>>(
        maxInputSize,
        outputSize,
        weights,
        biases,
        inputs,
        outputs
    );
}

extern "C" void singleSparseAffineBackward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const Feat* inputs,
    const float* errors)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    SingleSparseAffineBackwardKernel<<<grid, threads>>>(
        maxInputSize,
        outputSize,
        weightsGrad,
        biasesGrad,
        inputs,
        errors
    );
}

extern "C" void sparseAffineForward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const Feat* inputs,
    float* outputs)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    sparseAffineForwardKernel<<<grid, threads>>>(
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
    const Feat* inputs,
    const float* errors)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    sparseAffineBackwardKernel<<<grid, threads>>>(
        maxInputSize,
        outputSize,
        weightsGrad,
        biasesGrad,
        inputs,
        errors
    );
}
