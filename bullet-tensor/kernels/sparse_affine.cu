#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

__global__ void __kernel_sparse_affine_forward(
    const size_t inputSize,
    const size_t outputSize,
    const size_t half,
    const float* weights,
    const float* biases,
    const uint16_t* inputs,
    float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const uint16_t* thisInput = inputs + inputSize * blockIdx.y;
    float* thisOutput = outputs + 2 * outputSize * blockIdx.y + half + elem;

    float elementVal = biases[elem];

    for (size_t i = 0; i < inputSize; i++) {
        const size_t inp = static_cast<size_t>(thisInput[i]);

        if (inp == static_cast<size_t>(65535))
            break;

        const size_t idx = inp * outputSize + elem;
        elementVal += weights[idx];
    }

    thisOutput[0] = elementVal;
}

__global__ void __kernel_sparse_affine_backward(
    const size_t inputSize,
    const size_t outputSize,
    const size_t half,
    float* weightsGrad,
    float* biasesGrad,
    const uint16_t* inputs,
    const float* errors)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const uint16_t* thisInput = inputs + inputSize * blockIdx.y;
    const float* thisErrors = errors + 2 * outputSize * blockIdx.y + half;

    const float error = thisErrors[elem];

    atomicAdd(&biasesGrad[elem], error);

    for (size_t i = 0; i < inputSize; i++) {
        const size_t inp = static_cast<size_t>(thisInput[i]);

        if (inp == static_cast<size_t>(65535))
            break;

        const size_t idx = inp * outputSize + elem;
        atomicAdd(&weightsGrad[idx], error);
    }
}

extern "C" void sparseAffineForward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    const size_t half,
    const float* weights,
    const float* biases,
    const uint16_t* inputs,
    float* outputs)
{
    const size_t numChunks = static_cast<size_t>(1) + outputSize / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    __kernel_sparse_affine_forward<<<grid, threads>>>(
        maxInputSize,
        outputSize,
        half,
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
    const size_t half,
    float* weightsGrad,
    float* biasesGrad,
    const uint16_t* inputs,
    const float* errors)
{
    const size_t numChunks = static_cast<size_t>(1) + outputSize / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    __kernel_sparse_affine_backward<<<grid, threads>>>(
        maxInputSize,
        outputSize,
        half,
        weightsGrad,
        biasesGrad,
        inputs,
        errors
    );
}
