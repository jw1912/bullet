#include "util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

__global__ void selectKernel(
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    const int32_t* buckets,
    const float* in,
    float* out)
{
    const size_t thisIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thisIdx >= batchSize)
        return;

    const size_t thisBucket = static_cast<size_t>(buckets[thisIdx]);

    const float* thisInput = in + inputSize * thisIdx + outputSize * thisBucket;
    float* thisOutput = out + outputSize * thisIdx;

    for (size_t i = 0; i < outputSize; i++)
        thisOutput[i] = thisInput[i];
}

__global__ void selectBackpropKernel(
    const size_t batch_size,
    const size_t input_size,
    const size_t output_size,
    const int32_t* buckets,
    const float* output_grad,
    float* input_grad)
{
    const size_t thisIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thisIdx >= batch_size)
        return;

    const size_t thisBucket = static_cast<size_t>(buckets[thisIdx]);

    const float* thisOutputGrad = output_grad + output_size * thisIdx;
    float* thisInputGrad = input_grad + input_size * thisIdx + output_size * thisBucket;

    for (size_t i = 0; i < output_size; i++)
        thisInputGrad[i] += thisOutputGrad[i];
}

extern "C" void selectForward(
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    const int32_t* buckets,
    const float* in,
    float* out)
{
    const size_t numChunks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    const size_t threads = (numChunks == 1) ? batchSize : threadsPerBlock;

    selectKernel<<<numChunks, threads>>>(
        batchSize,
        inputSize,
        outputSize,
        buckets,
        in,
        out
    );
}

extern "C" void selectBackprop(
    const size_t batch_size,
    const size_t input_size,
    const size_t output_size,
    const int32_t* buckets,
    const float* output_grad,
    float* input_grad)
{
    const size_t numChunks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    const size_t threads = (numChunks == 1) ? batch_size : threadsPerBlock;

    selectBackpropKernel<<<numChunks, threads>>>(
        batch_size,
        input_size,
        output_size,
        buckets,
        output_grad,
        input_grad
    );
}
