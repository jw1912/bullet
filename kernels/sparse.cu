#include <iostream>
#include "util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

template<bool bias>
__global__ void sparseAffineForwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const int32_t* inputs,
    float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const size_t inputIdx = inputSize * blockIdx.y;
    const int32_t* thisInput = inputs + inputIdx;
    float* thisOutput = outputs + outputSize * blockIdx.y + elem;

    float ourElementVal;
    if constexpr (bias)
        ourElementVal = biases[elem];
    else
        ourElementVal = 0.0F;

    for (size_t i = 0; i < inputSize; i++) {
        const int32_t inp = thisInput[i];

        if (inp == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(inp) * outputSize + elem;
        ourElementVal += weights[ourIdx];
    }

    thisOutput[0] = ourElementVal;
}

template<bool bias>
__global__ void sparseAffineBackwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const int32_t* inputs,
    const float* errors)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const int32_t* thisInput = inputs + inputSize * blockIdx.y;
    const float* thisErrors = errors + outputSize * blockIdx.y;

    const float ourError = thisErrors[elem];

    if constexpr (bias)
        atomicAdd(&biasesGrad[elem], ourError);

    for (size_t i = 0; i < inputSize; i++) {
        const int32_t inp = thisInput[i];

        if (inp == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(inp) * outputSize + elem;
        atomicAdd(&weightsGrad[ourIdx], ourError);
    }
}

extern "C" void sparseAffineForward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const int32_t* inputs,
    float* outputs)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    if (biases == nullptr)
        sparseAffineForwardKernel<false><<<grid, threads>>>(maxInputSize, outputSize, weights, biases, inputs, outputs);
    else
        sparseAffineForwardKernel<true><<<grid, threads>>>(maxInputSize, outputSize, weights, biases, inputs, outputs);
}

extern "C" void sparseAffineBackward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const int32_t* inputs,
    const float* errors)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    if (biasesGrad == nullptr)
        sparseAffineBackwardKernel<false><<<grid, threads>>>(maxInputSize, outputSize, weightsGrad, biasesGrad, inputs, errors);
    else
        sparseAffineBackwardKernel<true><<<grid, threads>>>(maxInputSize, outputSize, weightsGrad, biasesGrad, inputs, errors);
}

typedef float(*OpType)(float);

template<OpType op>
__global__ void sparseAffineDualForwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const int32_t* stm,
    const int32_t* ntm,
    float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const size_t inputIdx = inputSize * blockIdx.y;
    const int32_t* thisStmInput = stm + inputIdx;
    const int32_t* thisNtmInput = ntm + inputIdx;
    float* thisOutput = outputs + 2 * outputSize * blockIdx.y + elem;

    float stmElementVal = biases[elem];
    float ntmElementVal = stmElementVal;

    for (size_t i = 0; i < inputSize; i++) {
        const int32_t stmInp = thisStmInput[i];
        const int32_t ntmInp = thisNtmInput[i];

        if (stmInp == -1 || ntmInp == -1)
            break;

        const size_t stmIdx = static_cast<size_t>(stmInp) * outputSize + elem;
        stmElementVal += weights[stmIdx];

        const size_t ntmIdx = static_cast<size_t>(ntmInp) * outputSize + elem;
        ntmElementVal += weights[ntmIdx];
    }

    thisOutput[0] = op(stmElementVal);
    thisOutput[outputSize] = op(ntmElementVal);
}

template<OpType op>
__global__ void sparseAffineDualBackwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const int32_t* stm,
    const int32_t* ntm,
    const float* outputs,
    const float* errors)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const int32_t* thisStmInput = stm + inputSize * blockIdx.y;
    const int32_t* thisNtmInput = ntm + inputSize * blockIdx.y;
    const float* thisOutputs = outputs + 2 * outputSize * blockIdx.y;
    const float* thisErrors = errors + 2 * outputSize * blockIdx.y;

    const float stmError = op(thisOutputs[elem]) * thisErrors[elem];
    const float ntmError = op(thisOutputs[elem + outputSize]) *thisErrors[elem + outputSize];

    atomicAdd(&biasesGrad[elem], stmError + ntmError);

    for (size_t i = 0; i < inputSize; i++) {
        const int32_t stmInp = thisStmInput[i];
        const int32_t ntmInp = thisNtmInput[i];

        if (stmInp == -1 || ntmInp == -1)
            break;

        const size_t stmIdx = static_cast<size_t>(stmInp) * outputSize + elem;
        atomicAdd(&weightsGrad[stmIdx], stmError);

        const size_t ntmIdx = static_cast<size_t>(ntmInp) * outputSize + elem;
        atomicAdd(&weightsGrad[ntmIdx], ntmError);
    }
}

extern "C" void sparseAffineDualForward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const int32_t* stm,
    const int32_t* ntm,
    float* outputs,
    const int32_t activation)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    switch (activation)
    {
        case 0:
            sparseAffineDualForwardKernel<Identity><<<grid, threads>>>(maxInputSize, outputSize, weights, biases, stm, ntm, outputs);
            break;
        case 1:
            sparseAffineDualForwardKernel<ReLU><<<grid, threads>>>(maxInputSize, outputSize, weights, biases, stm, ntm, outputs);
            break;
        case 2:
            sparseAffineDualForwardKernel<CReLU><<<grid, threads>>>(maxInputSize, outputSize, weights, biases, stm, ntm, outputs);
            break;
        case 3:
            sparseAffineDualForwardKernel<SCReLU><<<grid, threads>>>(maxInputSize, outputSize, weights, biases, stm, ntm, outputs);
            break;
        case 4:
            sparseAffineDualForwardKernel<SqrReLU><<<grid, threads>>>(maxInputSize, outputSize, weights, biases, stm, ntm, outputs);
            break;
        case 5:
            sparseAffineDualForwardKernel<sigmoid><<<grid, threads>>>(maxInputSize, outputSize, weights, biases, stm, ntm, outputs);
            break;
        default:
            std::cout << "Invalid activation function!" << std::endl;
            std::abort();
    }
}

extern "C" void sparseAffineDualBackward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const int32_t* stm,
    const int32_t* ntm,
    const float* outputs,
    const float* errors,
    const int32_t activation)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    switch (activation)
    {
        case 0:
            sparseAffineDualBackwardKernel<primeInvIdentity><<<grid, threads>>>(maxInputSize, outputSize, weightsGrad, biasesGrad, stm, ntm, outputs, errors);
            break;
        case 1:
            sparseAffineDualBackwardKernel<primeInvReLU><<<grid, threads>>>(maxInputSize, outputSize, weightsGrad, biasesGrad, stm, ntm, outputs, errors);
            break;
        case 2:
            sparseAffineDualBackwardKernel<primeInvCReLU><<<grid, threads>>>(maxInputSize, outputSize, weightsGrad, biasesGrad, stm, ntm, outputs, errors);
            break;
        case 3:
            sparseAffineDualBackwardKernel<primeInvSCReLU><<<grid, threads>>>(maxInputSize, outputSize, weightsGrad, biasesGrad, stm, ntm, outputs, errors);
            break;
        case 4:
            sparseAffineDualBackwardKernel<primeInvSqrReLU><<<grid, threads>>>(maxInputSize, outputSize, weightsGrad, biasesGrad, stm, ntm, outputs, errors);
            break;
        case 5:
            sparseAffineDualBackwardKernel<primeInvSigmoid><<<grid, threads>>>(maxInputSize, outputSize, weightsGrad, biasesGrad, stm, ntm, outputs, errors);
            break;
        default:
            std::cout << "Invalid activation function!" << std::endl;
            std::abort();
    }
}
