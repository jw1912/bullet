#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

struct Feat {
    uint16_t our;
    uint16_t opp;
};

__global__ void __kernel_sparse_affine_forward(
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

__global__ void __kernel_sparse_affine_backward(
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

extern "C" void sparseAffineForward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const Feat* inputs,
    float* outputs)
{
    const size_t numChunks = static_cast<size_t>(1) + outputSize / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    __kernel_sparse_affine_forward<<<grid, threads>>>(
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
    const size_t numChunks = static_cast<size_t>(1) + outputSize / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    __kernel_sparse_affine_backward<<<grid, threads>>>(
        maxInputSize,
        outputSize,
        weightsGrad,
        biasesGrad,
        inputs,
        errors
    );
}
