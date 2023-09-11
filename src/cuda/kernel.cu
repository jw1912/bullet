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

__global__ void populateAccumulator(
    const size_t batchSize,
    const size_t hiddenSize,
    const size_t inputSize,
    const float* featureWeights,
    const float* featureBiases,
    const uint16_t* inputs,
    float* accumulators)
{
    if (blockIdx.x >= batchSize)
        return;

    if (threadIdx.x >= hiddenSize)
        return;

    const size_t inputIdx = inputSize * blockIdx.x;
    const size_t element = threadIdx.x;
    const size_t outputIdx = hiddenSize * blockIdx.x + element;

    const uint16_t* thisInput = inputs + inputIdx;

    float elementVal = featureBiases[element];

    for (int i = 0; i < inputSize; i++) {
        if (thisInput[i] >= 768)
            break;

        const size_t idx = thisInput[i] * hiddenSize + element;
        elementVal += featureWeights[idx];
    }

    elementVal = elementVal < 0 ? 0 : elementVal > 1 ? 1 : elementVal;

    accumulators[outputIdx] = elementVal;
}

__global__ void setOutputBias(
    const size_t batchSize,
    const float* outputBias,
    float* outputs)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batchSize)
        return;

    outputs[idx] = outputBias[0];
}

__global__ void calculateEvals(
    const size_t batchSize,
    const size_t hiddenSize,
    const float* outputWeights,
    const float* outputBiases,
    const float* ourAccumulators,
    const float* oppAccumulators,
    float* outputs)
{
    if (blockIdx.x >= batchSize)
        return;

    if (threadIdx.x >= hiddenSize)
        return;

    const size_t element = threadIdx.x;
    const size_t outputIdx = blockIdx.x;
    const size_t idx = outputIdx * hiddenSize + element;

    float outputVal = ourAccumulators[idx] * outputWeights[element];
    outputVal += oppAccumulators[idx] * outputWeights[hiddenSize + element];

    atomicAdd(&outputs[outputIdx], outputVal);
}

__global__ void calculateErrors(
    const size_t batchSize,
    const size_t hiddenSize,
    const float* results,
    float* outputs,
    float* error)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batchSize)
        return;

    const float eval = outputs[idx];
    const float result = results[idx];
    const float sigmoid = 1.0 / (1.0 + expf(-eval));
    const float diff = sigmoid - result;
    const float singleError = diff * sigmoid * (1.0 - sigmoid);

    atomicAdd(error, diff * diff);

    outputs[idx] = singleError;
}

__global__ void backpropSide(
    const size_t batchSize,
    const size_t hiddenSize,
    const size_t inputSize,
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

    if (threadIdx.x >= hiddenSize)
        return;

    const size_t element = threadIdx.x;
    const size_t outputIdx = blockIdx.x;
    const size_t inputIdx = outputIdx * inputSize;
    const size_t outputWeightIdx = element + outputOffset;
    const size_t accumulatorIdx = outputIdx * hiddenSize + element;

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

    for (int i = 0; i < inputSize; i++) {
        if (thisInput[i] >= 768)
            break;

        const size_t x = thisInput[i] * hiddenSize + element;
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
        const size_t accumulatorSize = batchSize * hiddenSize * sizeof(float);
        const size_t outputSize = batchSize * sizeof(float);
        const size_t blocks = (batchSize + hiddenSize - 1) / hiddenSize;

        float* ourAccumulators;
        cudaMalloc(&ourAccumulators, accumulatorSize);
        cudaDeviceSynchronize();
        checkError("malloc 1");

        populateAccumulator<<<batchSize, hiddenSize>>>(batchSize, hiddenSize, inputSize, featureWeights, featureBiases, ourInputs, ourAccumulators);
        cudaDeviceSynchronize();
        checkError("accumulator 1");

        float* oppAccumulators;
        cudaMalloc(&oppAccumulators, accumulatorSize);
        cudaDeviceSynchronize();
        checkError("malloc 2");

        populateAccumulator<<<batchSize, hiddenSize>>>(batchSize, hiddenSize, inputSize, featureWeights, featureBiases, oppInputs, oppAccumulators);
        cudaDeviceSynchronize();
        checkError("accumulator 2 ");

        float* outputs;
        cudaMallocManaged(&outputs, outputSize);
        cudaDeviceSynchronize();
        checkError("malloc 3");

        setOutputBias<<<blocks, hiddenSize>>>(batchSize, outputBiases, outputs);
        cudaDeviceSynchronize();
        checkError("memset");

        calculateEvals<<<batchSize, hiddenSize>>>(batchSize, hiddenSize, outputWeights, outputBiases, ourAccumulators, oppAccumulators, outputs);
        cudaDeviceSynchronize();
        checkError("eval");

        calculateErrors<<<blocks, hiddenSize>>>(batchSize, hiddenSize, results, outputs, error);
        cudaDeviceSynchronize();
        checkError("error");

        backpropSide<<<batchSize, hiddenSize>>>(
            batchSize, hiddenSize, inputSize, 0,
            outputWeights, ourAccumulators, ourInputs, outputs,
            featureWeightsGradient, featureBiasesGradient, outputWeightsGradient
        );
        cudaDeviceSynchronize();
        checkError("backprops 1");

        backpropSide<<<batchSize, hiddenSize>>>(
            batchSize, hiddenSize, inputSize, hiddenSize,
            outputWeights, oppAccumulators, oppInputs, outputs,
            featureWeightsGradient, featureBiasesGradient, outputWeightsGradient
        );
        cudaDeviceSynchronize();
        checkError("backprops 2");

        backpropOutputBias<<<1, 1>>>(batchSize, outputs, outputBiasesGradient);
        cudaDeviceSynchronize();
        checkError("backprops 3");

        cudaFree(ourAccumulators);
        cudaFree(oppAccumulators);
        cudaFree(outputs);
        cudaDeviceSynchronize();
        checkError("freeing");

        return cudaGetLastError();
    }
}