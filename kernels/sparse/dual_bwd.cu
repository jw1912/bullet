#include <iostream>
#include "../util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

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
