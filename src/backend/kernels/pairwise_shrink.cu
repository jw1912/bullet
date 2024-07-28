/*
Computes
N = len(input_vector)
output_vector = input_vector[:N] * input_vector[N:]
(and gradients thereof)
*/
#include <cuda.h>
#include <cuda_runtime.h>

constexpr size_t threadsPerBlock = static_cast<size_t>(1024);

__global__ void pairwiseShrinkForwardKernel(
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    const float* input,
    float* output) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < batchSize * outputSize; i += stride) {
        const size_t batchIdx = i / outputSize;
        const size_t outputIdx = i % outputSize;

        const float* batchInput = input + batchIdx * inputSize;
        float* batchOutput = output + batchIdx * outputSize;

        batchOutput[outputIdx] = batchInput[outputIdx] * batchInput[outputIdx + outputSize];
    }
}

__global__ void pairwiseShrinkBackwardKernel(
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    const float* gradOutput,
    float* gradInput) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < batchSize * inputSize; i += stride) {
        const size_t batchIdx = i / inputSize;
        const size_t inputIdx = i % inputSize;

        const float* batchGradOutput = gradOutput + batchIdx * outputSize;
        const float* batchInput = gradInput + batchIdx * inputSize;  // Using gradInput as input here
        float* batchGradInput = gradInput + batchIdx * inputSize;

        if (inputIdx < outputSize) {
            batchGradInput[inputIdx] = batchGradOutput[inputIdx] * batchInput[inputIdx + outputSize];
        } else {
            batchGradInput[inputIdx] = batchGradOutput[inputIdx - outputSize] * batchInput[inputIdx - outputSize];
        }
    }
}

extern "C" void pairwiseShrinkForward(
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    const float* input,
    float* output) {
    const size_t totalElements = batchSize * outputSize;
    const size_t blocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    pairwiseShrinkForwardKernel<<<blocks, threadsPerBlock>>>(
        batchSize, inputSize, outputSize, input, output);
}

extern "C" void pairwiseShrinkBackward(
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    const float* gradOutput,
    float* gradInput) {
    const size_t totalElements = batchSize * inputSize;
    const size_t blocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    pairwiseShrinkBackwardKernel<<<blocks, threadsPerBlock>>>(
        batchSize, inputSize, outputSize, gradOutput, gradInput);
}