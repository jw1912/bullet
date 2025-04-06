#ifndef BULLET_CUDA_UTILS
#define BULLET_CUDA_UTILS
#include "util.cu"
#endif

#define ACTIVATE(name, op)\
BULLET_KERNEL name(const int size, const float* in, float* out)\
{\
    buffer_operation_kernel<op>(size, in, out);\
}\

#define BACKPROP(name, op)\
BULLET_KERNEL name(const int size, const float* input, const float* output_grad, float* input_grad)\
{\
    buffer_backprop_kernel<op>(size, input, output_grad, input_grad);\
}\

template<OpType op>
BULLET_KERNEL_IMPL buffer_operation_kernel(const int size, const float* in, float* out)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        const float4 a = ((const float4 *)in)[tid];
        ((float4 *)out)[tid] = make_float4(op(a.x), op(a.y), op(a.z), op(a.w));
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int idx = 4 * tid + i;
            out[idx] = op(in[idx]);
        }
    }
}

template<OpType op>
BULLET_KERNEL_IMPL buffer_backprop_kernel(const int size, const float* input, const float* output_grad, float* input_grad)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        const float4 this_in = ((const float4 *)input)[tid];
        const float4 this_out_grad = ((const float4 *)output_grad)[tid];
        float4 curr_input_grad = ((const float4 *)input_grad)[tid];

        curr_input_grad.x += op(this_in.x) * this_out_grad.x;
        curr_input_grad.y += op(this_in.y) * this_out_grad.y;
        curr_input_grad.z += op(this_in.z) * this_out_grad.z;
        curr_input_grad.w += op(this_in.w) * this_out_grad.w;

        ((float4 *)input_grad)[tid] = curr_input_grad;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            input_grad[j] += op(input[j]) * output_grad[j];
        }
    }
}

ACTIVATE(ForwardReluKernel, ReLU)
ACTIVATE(ForwardCreluKernel, CReLU)
ACTIVATE(ForwardScreluKernel, SCReLU)
ACTIVATE(ForwardSqrReluKernel, SqrReLU)
ACTIVATE(ForwardSigmoidKernel, sigmoid)

BACKPROP(BackwardReluKernel, primeReLU)
BACKPROP(BackwardCreluKernel, primeCReLU)
BACKPROP(BackwardScreluKernel, primeSCReLU)
BACKPROP(BackwardSqrReluKernel, primeSqrReLU)
BACKPROP(BackwardSigmoidKernel, primeSigmoid)
