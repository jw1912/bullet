#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

#define INPUT 32
#define HIDDEN 768

__global__ void addInternal(const float* A, const float* B, float* C, int size)
{
    int i = threadIdx.x;

    if (i >= size)
        return;

    C[i] = A[i] + B[i];
}

__global__ void populateAccumulator(
    const size_t batchSize,
    const float* featureWeights,
    const float* featureBiases,
    const uint16_t* inputs,
    float* accumulators)
{
    if (blockIdx.x >= batchSize)
        return;

    if (threadIdx.x >= HIDDEN)
        return;

    const size_t inputIdx = INPUT * blockIdx.x;
    const size_t element = threadIdx.x;
    const size_t outputIdx = HIDDEN * blockIdx.x + element;

    const uint16_t* thisInput = inputs + inputIdx;

    float elementVal = featureBiases[element];

    for (size_t i = 0; i < INPUT; i++) {
        if (thisInput[i] >= static_cast<uint16_t>(768))
            break;

        const size_t idx = static_cast<size_t>(thisInput[i]) * HIDDEN + element;
        elementVal += featureWeights[idx];
    }

    if (elementVal < 0)
        elementVal = 0;
    else if (elementVal > 1)
        elementVal = 1;

    accumulators[outputIdx] = elementVal;
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

    const float sigmoid = 1.0 / (1.0 + expf(-eval));
    const float diff = sigmoid - results[outputIdx];
    const float singleError = diff * sigmoid * (1.0 - sigmoid);

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

    if (threadIdx.x >= HIDDEN)
        return;

    const size_t element = threadIdx.x;
    const size_t outputIdx = blockIdx.x;
    const size_t inputIdx = outputIdx * INPUT;
    const size_t outputWeightIdx = element + outputOffset;
    const size_t accumulatorIdx = outputIdx * HIDDEN + element;

    const uint16_t* thisInput = inputs + inputIdx;

    const float error = outputs[outputIdx];
    const float weight = outputWeights[outputWeightIdx];
    const float accumulatorVal = accumulator[accumulatorIdx];

    // uses a trick
    const float component = accumulatorVal > 0 && accumulatorVal < 1
        ? error * weight
        : 0;

    atomicAdd(&featureBiasesGradient[element], component);
    atomicAdd(&outputWeightsGradient[outputWeightIdx], error * accumulatorVal);

    for (int i = 0; i < INPUT; i++) {
        if (thisInput[i] >= static_cast<uint16_t>(768))
            break;

        const size_t x = thisInput[i] * HIDDEN + element;
        atomicAdd(&featureWeightsGradient[x], component);
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

void checkError(std::string message)
{
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << message << std::endl;
        std::cout << cudaGetErrorString(error) << std::endl;
    }
}

size_t calcBlocks(size_t total, size_t threads)
{
    return (total + threads - 1) / threads;
}

extern "C" {
    cudaError add(const float* A, const float* B, float* C, int size)
    {
        addInternal<<<1, 3>>>(A, B, C, size);

        return cudaGetLastError();
    }

    cudaError trainBatch(
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
        float* error)
    {
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

        const size_t accumulatorSize = batchSize * HIDDEN * sizeof(float);
        const size_t outputSize = batchSize * sizeof(float);
        const size_t blocks = calcBlocks(batchSize, HIDDEN);
        const size_t sumBlocks = calcBlocks(batchSize, 1024);

        float* ourAccumulators;
        cudaMalloc(&ourAccumulators, accumulatorSize);
        float* oppAccumulators;
        cudaMalloc(&oppAccumulators, accumulatorSize);
        cudaDeviceSynchronize();

        populateAccumulator<<<batchSize, HIDDEN>>>(batchSize, featureWeights, featureBiases, ourInputs, ourAccumulators);
        cudaDeviceSynchronize();

        populateAccumulator<<<batchSize, HIDDEN>>>(batchSize, featureWeights, featureBiases, oppInputs, oppAccumulators);
        cudaDeviceSynchronize();

        float* outputs;
        cudaMalloc(&outputs, outputSize);
        cudaDeviceSynchronize();

        calculateErrors<<<sumBlocks, 1024>>>(batchSize, outputWeights, outputBiases, ourAccumulators, oppAccumulators, results, outputs, error);
        cudaDeviceSynchronize();

        backpropSide<<<batchSize, HIDDEN>>>(
            batchSize, 0,
            outputWeights, ourAccumulators, ourInputs, outputs,
            featureWeightsGradient, featureBiasesGradient, outputWeightsGradient
        );
        cudaDeviceSynchronize();

        backpropSide<<<batchSize, HIDDEN>>>(
            batchSize, HIDDEN,
            outputWeights, oppAccumulators, oppInputs, outputs,
            featureWeightsGradient, featureBiasesGradient, outputWeightsGradient
        );
        cudaDeviceSynchronize();

        backpropOutputBias<<<sumBlocks, 1024>>>(batchSize, outputs, outputBiasesGradient);
        cudaDeviceSynchronize();

        cudaFree(ourAccumulators);
        cudaFree(oppAccumulators);
        cudaFree(outputs);
        cudaDeviceSynchronize();

        return cudaGetLastError();
    }
}