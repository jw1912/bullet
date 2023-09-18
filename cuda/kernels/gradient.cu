/*
Calculating the gradient for a batch.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

#include "util.h"

// Just bit-twiddle to get next highest power of 2.
constexpr size_t determineChunkSize(size_t size)
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

constexpr size_t ChunkSize = determineChunkSize(static_cast<size_t>(HIDDEN));
constexpr size_t NumChunks = static_cast<size_t>(HIDDEN) / ChunkSize;

__global__ void populateAccumulator(
    const size_t batchSize,
    const float* featureWeights,
    const float* featureBiases,
    const uint16_t* inputs,
    float* accumulators)
{
    if (blockIdx.x >= batchSize)
        return;

    if (threadIdx.x >= NumChunks)
        return;

    const size_t inputIdx = INPUT * blockIdx.x;
    const size_t chunk = ChunkSize * threadIdx.x;
    const size_t outputIdx = HIDDEN * blockIdx.x + chunk;
    const uint16_t* thisInput = inputs + inputIdx;
    float* thisAccumulator = accumulators + outputIdx;

    #pragma unroll
    for (size_t element = 0; element < ChunkSize; element++)
    {
        const size_t offset = chunk + element;

        float elementVal = featureBiases[offset];

        for (size_t i = 0; i < INPUT; i++) {
            if (thisInput[i] == static_cast<uint16_t>(65535))
                break;

            const size_t idx = static_cast<size_t>(thisInput[i]) * HIDDEN + offset;
            elementVal += featureWeights[idx];
        }

        elementVal = activate(elementVal);

        thisAccumulator[element] = elementVal;
    }
}

__global__ void calculateErrors(
    const size_t batchSize,
    const float* outputWeights,
    const float* outputBiases,
    const float* ourAccumulators,
    const float* oppAccumulators,
    const float* results,
    float* outputs,
    float* error)
{
    const size_t outputIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outputIdx >= batchSize)
        return;

    const size_t accumulatorIdx = outputIdx * HIDDEN;

    float eval = outputBiases[0];

    for (size_t i = 0; i < HIDDEN; i++)
        eval += ourAccumulators[accumulatorIdx + i] * outputWeights[i];;

    for (size_t i = 0; i < HIDDEN; i++)
        eval += oppAccumulators[accumulatorIdx + i] * outputWeights[HIDDEN + i];

    const float sigmoid = 1.0F / (1.0F + expf(-eval));
    const float diff = sigmoid - results[outputIdx];
    const float singleError = diff * sigmoid * (1.0F - sigmoid);

    atomicAdd(error, diff * diff);

    outputs[outputIdx] = singleError;
}

__global__ void backpropSide(
    const size_t batchSize,
    const size_t outputOffset,
    const float* outputWeights,
    const float* accumulator,
    const uint16_t* inputs,
    const float* outputs,
    float* featureWeightsGradient,
    float* featureBiasesGradient,
    float* outputWeightsGradient)
{
    if (blockIdx.x >= batchSize)
        return;

    if (threadIdx.x >= NumChunks)
        return;

    const size_t chunk = ChunkSize * threadIdx.x;
    const size_t outputIdx = blockIdx.x;
    const size_t inputIdx = outputIdx * INPUT;
    const size_t outputWeightIdx = chunk + outputOffset;
    const size_t accumulatorIdx = outputIdx * HIDDEN + chunk;

    const uint16_t* thisInput = inputs + inputIdx;

    #pragma unroll
    for (size_t element = 0; element < ChunkSize; element++)
    {
        const float error = outputs[outputIdx];
        const float weight = outputWeights[outputWeightIdx + element];
        const float accumulatorVal = accumulator[accumulatorIdx + element];

        // uses a trick
        const float component = prime(accumulatorVal) * error * weight;

        atomicAdd(&featureBiasesGradient[chunk + element], component);
        atomicAdd(&outputWeightsGradient[outputWeightIdx + element], error * accumulatorVal);

        for (int i = 0; i < INPUT; i++) {
            if (thisInput[i] == static_cast<uint16_t>(65535))
                break;

            const size_t x = thisInput[i] * HIDDEN + chunk + element;
            atomicAdd(&featureWeightsGradient[x], component);
        }
    }
}

__global__ void backpropOutputBias(
    const size_t batchSize,
    const float* outputs,
    float* outputBiasesGradient)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batchSize)
        return;

    atomicAdd(outputBiasesGradient, outputs[idx]);
}

extern "C" cudaError calcGradient(
    const size_t batchSize,
    const size_t hiddenSize,
    const size_t inputSize,
    const float* featureWeights,
    const float* featureBiases,
    const float* outputWeights,
    const float* outputBiases,
    const uint16_t* ourInputs,
    const uint16_t* oppInputs,
    const float* results,
    float* featureWeightsGradient,
    float* featureBiasesGradient,
    float* outputWeightsGradient,
    float* outputBiasesGradient,
    float* error,
    float* ourAccumulators,
    float* oppAccumulators,
    float* outputs)
{
    static_assert(HIDDEN % ChunkSize == 0,
        "Net of this size must be divisible by an appropriate power of 2.");

    if (inputSize != INPUT)
    {
        std::cout << "Incompatible input format.";
        exit(1);
    }

    if (hiddenSize != HIDDEN)
    {
        std::cout << "HIDDEN must be set to " << hiddenSize << " in src/cuda/kernel.cu";
        exit(1);
    }

    const size_t blocks = calcBlocks(batchSize, HIDDEN);
    const size_t sumBlocks = calcBlocks(batchSize, 1024);

    populateAccumulator<<<batchSize, NumChunks>>>(batchSize, featureWeights, featureBiases, ourInputs, ourAccumulators);

    populateAccumulator<<<batchSize, NumChunks>>>(batchSize, featureWeights, featureBiases, oppInputs, oppAccumulators);

    calculateErrors<<<sumBlocks, 1024>>>(batchSize, outputWeights, outputBiases, ourAccumulators, oppAccumulators, results, outputs, error);

    backpropSide<<<batchSize, NumChunks>>>(
        batchSize, 0,
        outputWeights, ourAccumulators, ourInputs, outputs,
        featureWeightsGradient, featureBiasesGradient, outputWeightsGradient
    );

    backpropSide<<<batchSize, NumChunks>>>(
        batchSize, HIDDEN,
        outputWeights, oppAccumulators, oppInputs, outputs,
        featureWeightsGradient, featureBiasesGradient, outputWeightsGradient
    );

    backpropOutputBias<<<sumBlocks, 1024>>>(batchSize, outputs, outputBiasesGradient);

    cudaDeviceSynchronize();

    return cudaGetLastError();
}
