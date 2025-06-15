#include "../util.cu"

__global__ void set_kernel(float* buf, int32_t size, float val)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) buf[tid] = val;
}

extern "C" void set(float* buf, size_t size, float val)
{
    const size_t threads = 512;
    const size_t blocks = (size + threads - 1) / threads;
    set_kernel<<<blocks, threads>>>(buf, size, val);
}

#define SCALAR_KERNEL_FORWARD(name, op)\
extern "C" void name(const size_t size, const float alpha, const float* in, float* out)\
{\
    const size_t threads = 512;\
    const size_t float4_size = (size + 3) / 4;\
    const size_t blocks = (float4_size + threads - 1) / threads;\
    scalar_kernel_forward<op><<<blocks, threads>>>(size, alpha, in, out);\
}\

#define SCALAR_KERNEL_BACKWARD(name, op)\
extern "C" void name(const size_t size, const float alpha, const float* input, const float* output_grad, float* input_grad)\
{\
    const size_t threads = 512;\
    const size_t float4_size = (size + 3) / 4;\
    const size_t blocks = (float4_size + threads - 1) / threads;\
    scalar_kernel_backward<op><<<blocks, threads>>>(size, alpha, input, output_grad, input_grad);\
}\

template<BinaryOpType op>
__global__ void scalar_kernel_forward(const int32_t size, const float alpha, const float* inp, float* out) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

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
        for (int32_t i = 0; i < size - 4 * tid; i++)
        {
            const int32_t j = 4 * tid + i;
            out[j] = op(inp[j], alpha);
        }
    }
}

template<BinaryOpType op>
__global__ void scalar_kernel_backward(const int32_t size, const float alpha, const float* input, const float* output_grad, float* input_grad) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

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
        for (int32_t i = 0; i < size - 4 * tid; i++)
        {
            const int32_t j = 4 * tid + i;
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

SCALAR_KERNEL_FORWARD(add_scalar, add)
SCALAR_KERNEL_FORWARD(abs_pow_scalar, abs_pow)
SCALAR_KERNEL_BACKWARD(abs_pow_scalar_backward, abs_pow_backward)
