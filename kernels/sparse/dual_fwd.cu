#include <iostream>
#include "../util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

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
