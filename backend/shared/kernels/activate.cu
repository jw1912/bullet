#include "util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

template<OpType op>
__global__ void buffer_operation_kernel(const size_t size, const float* in, float* out)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        const float4 a = ((const float4 *)in)[tid];
        ((float4 *)out)[tid] = make_float4(op(a.x), op(a.y), op(a.z), op(a.w));
    }
    else if (4 * tid < size)
    {
        for (size_t i = 0; i < size - 4 * tid; i++)
        {
            const size_t idx = 4 * tid + i;
            out[idx] = op(in[idx]);
        }
    }
}

template<OpType op>
void buffer_operation(const size_t size, const float* in, float* out)
{
    const size_t threads = 512;
    const size_t float4_size = (size + 3) / 4;
    const size_t blocks = (float4_size + threads - 1) / threads;
    buffer_operation_kernel<op><<<blocks, threads>>>(size, in, out);
}

template<OpType op>
__global__ void buffer_backprop_kernel(const size_t size, const float* input, const float* output_grad, float* input_grad)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        const float4 this_in = ((const float4 *)input)[tid];
        const float4 this_out_grad = ((const float4 *)output_grad)[tid];
        const float4 curr_input_grad = ((const float4 *)input_grad)[tid];

        ((float4 *)input_grad)[tid] = make_float4(
            curr_input_grad.x + op(this_in.x) * this_out_grad.x,
            curr_input_grad.y + op(this_in.y) * this_out_grad.y,
            curr_input_grad.z + op(this_in.z) * this_out_grad.z,
            curr_input_grad.w + op(this_in.w) * this_out_grad.w
        );
    }
    else if (4 * tid < size)
    {
        for (size_t i = 0; i < size - 4 * tid; i++)
        {
            const size_t idx = 4 * tid + i;
            const float this_in = input[i];
            const float this_out_grad = output_grad[i];
            input_grad[idx] += op(this_in) * this_out_grad;
        }
    }
}

template<OpType op>
void buffer_backprop(const size_t size, const float* input, const float* output_grad, float* input_grad)
{
    const size_t threads = 512;
    const size_t float4_size = (size + 3) / 4;
    const size_t blocks = (float4_size + threads - 1) / threads;
    buffer_backprop_kernel<op><<<blocks, threads>>>(size, input, output_grad, input_grad);
}

extern "C" {
    void backpropReLU(const size_t size, const float* input, const float* output_grad, float* input_grad)
    {
        buffer_backprop<primeReLU>(size, input, output_grad, input_grad);
    }

    void backpropCReLU(const size_t size, const float* input, const float* output_grad, float* input_grad)
    {
        buffer_backprop<primeCReLU>(size, input, output_grad, input_grad);
    }

    void backpropSCReLU(const size_t size, const float* input, const float* output_grad, float* input_grad)
    {
        buffer_backprop<primeSCReLU>(size, input, output_grad, input_grad);
    }

    void backpropSqrReLU(const size_t size, const float* input, const float* output_grad, float* input_grad)
    {
        buffer_backprop<primeSqrReLU>(size, input, output_grad, input_grad);
    }

    void backpropSigmoid(const size_t size, const float* input, const float* output_grad, float* input_grad)
    {
        buffer_backprop<primeSigmoid>(size, input, output_grad, input_grad);
    }

    void activateReLU(const size_t size, const float* in, float* out)
    {
        buffer_operation<ReLU>(size, in, out);
    }

    void activateCReLU(const size_t size, const float* in, float* out)
    {
        buffer_operation<CReLU>(size, in, out);
    }

    void activateSCReLU(const size_t size, const float* in, float* out)
    {
        buffer_operation<SCReLU>(size, in, out);
    }

    void activateSqrReLU(const size_t size, const float* in, float* out)
    {
        buffer_operation<SqrReLU>(size, in, out);
    }

    void activateSigmoid(const size_t size, const float* in, float* out)
    {
        buffer_operation<sigmoid>(size, in, out);
    }
}
