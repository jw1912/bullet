#ifndef BULLET_CUDA_UTILS
#define BULLET_CUDA_UTILS
#include "util.cu"
#endif

BULLET_KERNEL SetKernel(float* buf, int size, float val)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) buf[tid] = val;
}

#define SCALAR_KERNEL_FORWARD(name, op)\
BULLET_KERNEL name(const int size, const float alpha, const float* in, float* out)\
{\
    scalar_kernel_forward<op>(size, alpha, in, out);\
}\

#define SCALAR_KERNEL_BACKWARD(name, op)\
BULLET_KERNEL name(const int size, const float alpha, const float* input, const float* output_grad, float* input_grad)\
{\
    scalar_kernel_backward<op>(size, alpha, input, output_grad, input_grad);\
}\

template<BinaryOpType op>
BULLET_KERNEL_IMPL scalar_kernel_forward(const int size, const float alpha, const float* inp, float* out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        float4 a = ((float4 *)out)[tid];
        const float4 b = ((const float4 *)inp)[tid];
        
        a.x = op(b.x, alpha);
        a.y = op(b.y, alpha);
        a.z = op(b.z, alpha);
        a.w = op(b.w, alpha);

        ((float4 *)out)[tid] = a;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            out[j] = op(inp[j], alpha);
        }
    }
}

template<BinaryOpType op>
BULLET_KERNEL_IMPL scalar_kernel_backward(const int size, const float alpha, const float* input, const float* output_grad, float* input_grad) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        const float4 this_in = ((const float4 *)input)[tid];
        const float4 this_out_grad = ((const float4 *)output_grad)[tid];
        float4 curr_input_grad = ((const float4 *)input_grad)[tid];

        curr_input_grad.x += op(this_in.x, alpha) * this_out_grad.x;
        curr_input_grad.y += op(this_in.y, alpha) * this_out_grad.y;
        curr_input_grad.z += op(this_in.z, alpha) * this_out_grad.z;
        curr_input_grad.w += op(this_in.w, alpha) * this_out_grad.w;

        ((float4 *)input_grad)[tid] = curr_input_grad;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            input_grad[j] += op(input[j], alpha) * output_grad[j];
        }
    }
}

__device__ float add(float a, float b) { return a + b; }
__device__ float abs_pow(float a, float b) { return powf(abs(a), b); }
__device__ float abs_pow_backward(float a, float b) {
    const float grad = b * powf(abs(a), b - 1.0F);
    return a > 0.0F ? grad : -grad;
};

SCALAR_KERNEL_FORWARD(AddScalarKernel, add)
SCALAR_KERNEL_FORWARD(AbsPowScalarKernel, abs_pow)
SCALAR_KERNEL_BACKWARD(AbsPowScalarBackwardKernel, abs_pow_backward)
