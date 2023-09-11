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
__device__ void accumulatePerspective(
    float* featureWeights,
    float* featureBiases,
    const uint16_t* inputs,
    float* accumulators,
    const size_t batchSize)
{
    if (blockIdx.x >= batchSize)
        return;

    if (threadIdx.x >= hiddenSize)
        return;

    const size_t inputIdx = inputSize * blockIdx.x;
    const size_t element = threadIdx.x;
    const size_t outputIdx = blockIdx.x * hiddenSize + element;

    const std::uint16_t* thisInput = inputs + inputIdx;

    float elementVal = featureBiases[element];

    for (int i = 0; i < inputSize; i++) {
        if (thisInput[i] == std::numeric_limits<uint16_t>::max())
            break;

        elementVal += featureWeights[thisInput[i] * hiddenSize + element];
    }

    elementVal = elementVal < 0 ? 0 : elementVal > 1 ? 1 : elementVal;

    accumulators[outputIdx] = elementVal;
}

template<const size_t hiddenSize>
__device__ void eval(
    float* outputWeights,
    float* outputBiases,
    const uint16_t* ourAccumulators,
    const uint16_t* oppAccumulators,
    float* outputs,
    const size_t batchSize)
{
    if (blockIdx.x >= batchSize)
        return;

    if (threadIdx.x >= hiddenSize)
        return;

    const size_t outputIdx = blockIdx.x;
    const size_t element = threadIdx.x;
    const size_t idx = hiddenSize * outputIdx + element;

    float outputVal = outputBiases[element];
    outputVal += ourAccumulators[idx] * outputWeights[element];
    outputVal += oppAccumulators[idx] * outputWeights[hiddenSize + element];

    atomicAdd(&outputs[outputIdx], outputVal);
}

extern "C" {
    cudaError add(const float* A, const float* B, float* C, int size) {
        addInternal<<<1, 3>>>(A, B, C, size);

        return cudaGetLastError();
    }
}