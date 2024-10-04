#include "util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

typedef float(*OpType)(float);

template<OpType op>
__global__ void bufferOperation(const size_t size, const float* in, float* out)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    out[i] = op(in[i]);
}

template<OpType op>
__global__ void bufferBackprop(const size_t size, const float* input, const float* output_grad, float* input_grad)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    const float thisIn = input[i];
    const float thisOutGrd = output_grad[i];

    input_grad[i] = op(thisIn) * thisOutGrd;
}

extern "C" void backpropReLU(const size_t size, const float* input, const float* output_grad, float* input_grad)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferBackprop<primeReLU><<<numBlocks, threadsPerBlock>>>(size, input, output_grad, input_grad);
}

extern "C" void backpropCReLU(const size_t size, const float* input, const float* output_grad, float* input_grad)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferBackprop<primeCReLU><<<numBlocks, threadsPerBlock>>>(size, input, output_grad, input_grad);
}

extern "C" void backpropSCReLU(const size_t size, const float* input, const float* output_grad, float* input_grad)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferBackprop<primeSCReLU><<<numBlocks, threadsPerBlock>>>(size, input, output_grad, input_grad);
}

extern "C" void backpropSqrReLU(const size_t size, const float* input, const float* output_grad, float* input_grad)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferBackprop<primeSqrReLU><<<numBlocks, threadsPerBlock>>>(size, input, output_grad, input_grad);
}

extern "C" void backpropSigmoid(const size_t size, const float* output, const float* output_grad, float* input_grad)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferBackprop<primeSqrReLU><<<numBlocks, threadsPerBlock>>>(size, output, output_grad, input_grad);
}

extern "C" void activateReLU(const size_t size, const float* in, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferOperation<ReLU><<<numBlocks, threadsPerBlock>>>(size, in, out);
}

extern "C" void activateCReLU(const size_t size, const float* in, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferOperation<CReLU><<<numBlocks, threadsPerBlock>>>(size, in, out);
}

extern "C" void activateSCReLU(const size_t size, const float* in, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferOperation<SCReLU><<<numBlocks, threadsPerBlock>>>(size, in, out);
}

extern "C" void activateSqrReLU(const size_t size, const float* in, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferOperation<SqrReLU><<<numBlocks, threadsPerBlock>>>(size, in, out);
}

extern "C" void activateSigmoid(const size_t size, const float* in, float* out)
{
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    bufferOperation<sigmoid><<<numBlocks, threadsPerBlock>>>(size, in, out);
}
