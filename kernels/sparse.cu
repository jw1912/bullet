#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

__global__ void sparseLinearForwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    const float* weights,
    const int32_t* inputs,
    float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const size_t inputIdx = inputSize * blockIdx.y;
    const int32_t* thisInput = inputs + inputSize * blockIdx.y;
    float* thisOutput = outputs + outputSize * blockIdx.y + elem;

    float ourElementVal = 0;

    for (size_t i = 0; i < inputSize; i++) {
        const int32_t inp = thisInput[i];

        if (inp == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(inp) * outputSize + elem;
        ourElementVal += weights[ourIdx];
    }

    thisOutput[0] = ourElementVal;
}

__global__ void sparseLinearBackwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    float* weightsGrad,
    const int32_t* inputs,
    const float* errors)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const int32_t* thisInput = inputs + inputSize * blockIdx.y;
    const float* thisErrors = errors + outputSize * blockIdx.y;

    float ourError = thisErrors[elem];

    for (size_t i = 0; i < inputSize; i++) {
        const int32_t inp = thisInput[i];

        if (inp == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(inp) * outputSize + elem;
        atomicAdd(&weightsGrad[ourIdx], ourError);
    }
}

extern "C" void sparseLinearForward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    const float* weights,
    const int32_t* inputs,
    float* outputs)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    sparseLinearForwardKernel<<<grid, threads>>>(
        maxInputSize,
        outputSize,
        weights,
        inputs,
        outputs
    );
}

extern "C" void sparseLinearBackward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    float* weightsGrad,
    const int32_t* inputs,
    const float* errors)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    sparseLinearBackwardKernel<<<grid, threads>>>(
        maxInputSize,
        outputSize,
        weightsGrad,
        inputs,
        errors
    );
}
