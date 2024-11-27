#include "util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

__global__ void pairwiseMulKernel(const size_t output_size, const float* input, float* output)
{
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= output_size)
        return;

    const float* thisInp = input + 2 * output_size * blockIdx.y + tid;
    float* thisOut = output + output_size * blockIdx.y + tid;

    thisOut[0] = thisInp[0] * thisInp[output_size];
}

__global__ void pairwiseMulBackwardKernel(const size_t output_size, const float* input, const float* output_grad, float* input_grad)
{
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= output_size)
        return;

    const float* thisOutputGrad = output_grad + output_size * blockIdx.y + tid;
    
    const size_t inputOffset = 2 * output_size * blockIdx.y + tid;
    const float* thisInput = input + inputOffset;
    float* thisInputGrad = input_grad + inputOffset;

    const float gradIn = thisOutputGrad[0];

    thisInputGrad[0] += gradIn * thisInput[output_size];
    thisInputGrad[output_size] += gradIn * thisInput[0];
}

extern "C" void pairwiseMul(const size_t batch_size, const size_t output_size, const float* input, float* output)
{
    const size_t grid_x = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    const dim3 grid(grid_x, batch_size);

    pairwiseMulKernel<<<grid, threadsPerBlock>>>(output_size, input, output);
}

extern "C" void backpropPairwiseMul(const size_t batch_size, const size_t output_size, const float* input, const float* output_grad, float* input_grad)
{
    const size_t grid_x = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    const dim3 grid(grid_x, batch_size);

    pairwiseMulBackwardKernel<<<grid, threadsPerBlock>>>(output_size, input, output_grad, input_grad);
}