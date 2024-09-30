#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

__global__ void selectKernel(
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    const uint8_t* buckets,
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
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    const uint8_t* buckets,
    const float* in,
    float* out)
{
    const size_t thisIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thisIdx >= batchSize)
        return;

    const size_t thisBucket = static_cast<size_t>(buckets[thisIdx]);

    const float* thisInput = in + inputSize * thisIdx;
    float* thisOutput = out + outputSize * thisIdx + inputSize * thisBucket;

    for (size_t i = 0; i < inputSize; i++)
        thisOutput[i] = thisInput[i];
}

constexpr size_t Threads = static_cast<size_t>(1024);

extern "C" void selectForward(
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    const uint8_t* buckets,
    const float* in,
    float* out)
{
    const size_t numChunks = (batchSize + Threads - 1) / Threads;
    const size_t threads = (numChunks == 1) ? batchSize : Threads;

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
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    const uint8_t* buckets,
    const float* in,
    float* out)
{
    const size_t numChunks = (batchSize + Threads - 1) / Threads;
    const size_t threads = (numChunks == 1) ? batchSize : Threads;

    selectBackpropKernel<<<numChunks, threads>>>(
        batchSize,
        inputSize,
        outputSize,
        buckets,
        in,
        out
    );
}
