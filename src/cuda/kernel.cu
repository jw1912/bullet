#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

__global__ void addInternal(const float* A, const float* B, float* C, int size)
{
    int i = threadIdx.x;

    if (i >= size)
        return;

    C[i] = A[i] + B[i];
}

template<const size_t hiddenSize, const size_t inputSize>
__global__ void accumulatePerspective(
    float* featureWeights,
    float* featureBiases,
    const uint16_t* inputs,
    float* outputs,
    const size_t batchSize)
{
    const size_t inputIdx = inputSize * blockIdx.x;
    const size_t element = threadIdx.x;
    const size_t outputIdx = blockIdx.x * hiddenSize + element;

    if (inputIdx >= inputSize * batchSize)
        return;

    if (element >= hiddenSize)
        return;

    const std::uint16_t* thisInput = inputs + inputIdx;

    float elementVal = featureBiases[element];

    for (int i = 0; i < inputSize; i++) {
        if (thisInput[i] == std::numeric_limits<uint16_t>::max())
            break;

        elementVal += featureWeights[thisInput[i] * hiddenSize + element];
    }

    elementVal = elementVal < 0 ? 0 : elementVal > 1 ? 1 : elementVal;

    outputs[outputIdx] = elementVal;
}

extern "C" {
cudaError add(const float* A, const float* B, float* C, int size) {
    addInternal<<<1, 3>>>(A, B, C, size);

    return cudaGetLastError();
}
}