#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

__global__ void __kernel_sparse_affine_forward(
    const size_t chunkSize,
    const size_t inputSize,
    const size_t outputSize,
    const size_t half,
    const float* weights,
    const float* biases,
    const uint16_t* inputs,
    float* outputs)
{
    const size_t inputIdx = inputSize * blockIdx.x;
    const size_t chunk = chunkSize * threadIdx.x;

    // 2 perspectives
    const size_t outputIdx = 2 * outputSize * blockIdx.x + chunk;

    const uint16_t* thisInput = inputs + inputIdx;
    float* thisAccumulator = outputs + outputIdx;

    for (size_t element = 0; element < chunkSize; element++)
    {
        const size_t offset = chunk + element;

        if (offset >= outputSize)
            return;

        float elementVal = biases[offset];

        for (size_t i = 0; i < inputSize; i++) {
            const size_t inp = static_cast<size_t>(thisInput[i]);

            if (inp == static_cast<size_t>(65535))
                break;

            const size_t idx = inp * outputSize + offset;
            elementVal += weights[idx];
        }

        thisAccumulator[half + element] = elementVal;
    }
}

__global__ void __kernel_sparse_affine_backward(
    const size_t chunkSize,
    const size_t inputSize,
    const size_t outputSize,
    const size_t half,
    float* weightsGrad,
    float* biasesGrad,
    const uint16_t* inputs,
    const float* errors)
{
    const size_t inputIdx = inputSize * blockIdx.x;
    const size_t chunk = chunkSize * threadIdx.x;

    // two perspectives
    const size_t outputIdx = half + 2 * outputSize * blockIdx.x + chunk;

    const uint16_t* thisInput = inputs + inputIdx;

    for (size_t element = 0; element < chunkSize; element++)
    {
        const size_t offset = chunk + element;

        if (offset >= outputSize)
            return;

        const float error = errors[outputIdx + element];
        atomicAdd(&biasesGrad[offset], error);

        for (size_t i = 0; i < inputSize; i++) {
            const size_t inp = static_cast<size_t>(thisInput[i]);

            if (inp == static_cast<size_t>(65535))
                break;

            const size_t idx = inp * outputSize + offset;
            atomicAdd(&weightsGrad[idx], error);
        }
    }
}

size_t determineChunkSize(size_t size)
{
    size--;
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    size |= size >> 16;
    size |= size >> 32;
    size++;

    const size_t chunkSize = size / 1024;

    return chunkSize == 0 ? 1 : chunkSize;
}

extern "C" void sparseAffineForward(
    const size_t batchSize,
    const size_t chunkSize,
    const size_t maxInputSize,
    const size_t outputSize,
    const size_t half,
    const float* weights,
    const float* biases,
    const uint16_t* inputs,
    float* outputs)
{
    const size_t numChunks = (outputSize + chunkSize - 1) / chunkSize;

    __kernel_sparse_affine_forward<<<batchSize, numChunks>>>(
        chunkSize,
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
    const size_t chunkSize,
    const size_t maxInputSize,
    const size_t outputSize,
    const size_t half,
    float* weightsGrad,
    float* biasesGrad,
    const uint16_t* inputs,
    const float* errors)
{
    const size_t numChunks = (outputSize + chunkSize - 1) / chunkSize;

    __kernel_sparse_affine_backward<<<batchSize, numChunks>>>(
        chunkSize,
        maxInputSize,
        outputSize,
        half,
        weightsGrad,
        biasesGrad,
        inputs,
        errors
    );
}
