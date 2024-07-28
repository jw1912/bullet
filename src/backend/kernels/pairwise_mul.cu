/*
Computes
N = len(input_vector)
output_vector = input_vector[:N] * input_vector[N:]
(and gradients thereof)
*/
#include <cuda.h>
#include <cuda_runtime.h>

// This file is commented to death, because it was written by someone who doesn't know CUDA very well (cosmo).

constexpr size_t threadsPerBlock = static_cast<size_t>(1024);

__global__ void pairwiseMulForwardKernel(
    const size_t batchSize,
    // when going FORWARD, input is twice output
    const size_t inputSize,
    // when going FORWARD, output is half input
    const size_t outputSize,
    const float* input,
    float* output) {
    // Calculate the global thread ID
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the stride for grid-stride loop
    // This allows handling more elements than available threads
    const size_t stride = blockDim.x * gridDim.x;

    // Grid-stride loop: process elements with a stride
    for (size_t i = tid; i < batchSize * outputSize; i += stride) {
        // Calculate batch index and output index within the batch
        const size_t batchIdx = i / outputSize;
        const size_t outputIdx = i % outputSize;

        // Calculate pointers for the current batch
        const float* batchInput = input + batchIdx * inputSize;
        float* batchOutput = output + batchIdx * outputSize;

        // Perform pairwise multiplication:
        // Multiply element from first half with corresponding element from second half
        // INVARIANT: outputSize is always half of inputSize.
        batchOutput[outputIdx] = batchInput[outputIdx] * batchInput[outputIdx + outputSize];
    }
}

__global__ void pairwiseMulBackwardKernel(
    const size_t batchSize,
    // when going BACKWARD, input is half output
    const size_t inputSize,
    // when going BACKWARD, output is twice input
    const size_t outputSize,
    // gradients on the output neurons
    const float* input,
    // buffer to write gradients into
    float* output) {
    // Calculate the global thread ID
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the stride for grid-stride loop
    const size_t stride = blockDim.x * gridDim.x;

    // Grid-stride loop: process elements with a stride
    for (size_t i = tid; i < batchSize * inputSize; i += stride) {
        // Calculate batch index and input index within the batch
        const size_t batchIdx = i / inputSize;
        const size_t outputIdx = i % inputSize;

        // Calculate pointers for the current batch
        const float* batchGradOutput = input + batchIdx * inputSize;
        float* batchGradInput = output + batchIdx * outputSize;

        // Compute gradients
        // NOTE: we need to do _both_ of these reads before we do the writes,
        // because we would corrupt the computations if we interleaved them.
        const float gradLeft = batchGradOutput[outputIdx] * batchGradInput[outputIdx + inputSize];
        const float gradRight = batchGradOutput[outputIdx] * batchGradInput[outputIdx];

        batchGradInput[outputIdx] = gradLeft;
        batchGradInput[outputIdx + inputSize] = gradRight;
    }
}

extern "C" void pairwiseMul(
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    const float* input,
    float* output) {
    // Calculate total number of elements to process
    const size_t totalElements = batchSize * outputSize;
    // Calculate number of blocks needed
    const size_t blocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    pairwiseMulForwardKernel<<<blocks, threadsPerBlock>>>(
        batchSize, inputSize, outputSize, input, output);
}

extern "C" void backpropPairwiseMul(
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    // gradients on the output
    const float* input,
    // buffer to write gradients into
    float* output) {
    // Calculate total number of elements to process
    const size_t totalElements = batchSize * inputSize;
    // Calculate number of blocks needed
    const size_t blocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    pairwiseMulBackwardKernel<<<blocks, threadsPerBlock>>>(
        batchSize, inputSize, outputSize, input, output);
}