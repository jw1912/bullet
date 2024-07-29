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

__global__ void pairwiseMulKernel(
    const size_t batchSize,
    const size_t tensorSize,
    const float* inp,
    float* out) {
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= tensorSize)
        return;

    // input vector is 2x size   vvvvvvvvvvvvvv
    const float* thisInp = inp + 2 * tensorSize * blockIdx.y + tid;
    float* thisOut = out + tensorSize * blockIdx.y + tid;

    thisOut[0] = thisInp[0] * thisInp[tensorSize];
}

extern "C" void pairwiseMul(
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    const float* input,
    float* output) {
    const size_t grid_x = (outputSize + threadsPerBlock - 1) / threadsPerBlock;
    const dim3 grid(grid_x, batchSize);

    pairwiseMulKernel<<<grid, threadsPerBlock>>>(
        batchSize,
        outputSize,
        inp,
        out);
}

__global__ void pairwiseMulBackwardKernel(
    const size_t batchSize,
    const size_t tensorSize,
    const float* inp,
    float* out) {
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= tensorSize)
        return;

    const float* thisInp = inp + tensorSize * blockIdx.y + tid;
    float* thisOut = out + 2 * tensorSize * blockIdx.y + tid;

    // thisOut[0] = thisInp[0] * thisInp[tensorSize];
    const float gradIn = thisInp[0];
    const float valLeft = thisOut[0];
    const float valRight = thisOut[tensorSize];
    const float gradLeft = gradIn * valRight;
    const float gradRight = gradIn * valLeft;

    thisOut[0] = gradLeft;
    thisOut[tensorSize] = gradRight;
}

extern "C" void backpropPairwiseMul(
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    // gradients on the output
    const float* input,
    // buffer to write gradients into
    float* output) {
    const size_t grid_x = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
    const dim3 grid(grid_x, batchSize);

    // Launch the kernel
    pairwiseMulBackwardKernel<<<grid, threadsPerBlock>>>(
        batchSize, inputSize, input, output);
}