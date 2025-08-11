#include "util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

__global__ void selectKernel(
    const size_t batch_size,
    const int32_t input_batched,
    const size_t input_size,
    const size_t output_size,
    const int32_t* buckets,
    const float* in,
    float* out)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= batch_size * output_size)
        return;

    const size_t idxInBatch = tid / output_size;
    const size_t idxInOutput = tid % output_size;

    const size_t thisBucket = static_cast<size_t>(buckets[idxInBatch]);

    const float thisInput = in[output_size * thisBucket + idxInOutput + (input_batched ? input_size * idxInBatch : 0)];
    out[output_size * idxInBatch + idxInOutput] = thisInput;
}

__global__ void selectBackpropKernel(
    const size_t batch_size,
    const int32_t input_grad_batched,
    const size_t input_size,
    const size_t output_size,
    const int32_t* buckets,
    const float* output_grad,
    float* input_grad)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= batch_size * output_size)
        return;

    const size_t idxInBatch = tid / output_size;
    const size_t idxInOutput = tid % output_size;

    const size_t thisBucket = static_cast<size_t>(buckets[idxInBatch]);

    const float thisOutputGrad = output_grad[output_size * idxInBatch + idxInOutput];
    float* thisInputGrad = input_grad + output_size * thisBucket + idxInOutput;

    if (input_grad_batched)
    {
        thisInputGrad[input_size * idxInBatch] += thisOutputGrad;
    }
    else
    {
        atomicAdd(thisInputGrad, thisOutputGrad);
    }
}

extern "C" void selectForward(
    const size_t batch_size,
    const int32_t input_batched,
    const size_t input_size,
    const size_t output_size,
    const int32_t* buckets,
    const float* in,
    float* out)
{
    const size_t blocks = (batch_size * output_size + threadsPerBlock - 1) / threadsPerBlock;

    selectKernel<<<blocks, threadsPerBlock>>>(
        batch_size,
        input_batched,
        input_size,
        output_size,
        buckets,
        in,
        out
    );
}

extern "C" void selectBackprop(
    const size_t batch_size,
    const int32_t input_grad_batched,
    const size_t input_size,
    const size_t output_size,
    const int32_t* buckets,
    const float* output_grad,
    float* input_grad)
{
    const size_t blocks = (batch_size * output_size + threadsPerBlock - 1) / threadsPerBlock;

    selectBackpropKernel<<<blocks, threadsPerBlock>>>(
        batch_size,
        input_grad_batched,
        input_size,
        output_size,
        buckets,
        output_grad,
        input_grad
    );
}
