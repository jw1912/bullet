#include "../util.cu"

#define ACTIVATE(name, op)\
extern "C" void name(const size_t size, const float* in, float* out)\
{\
    const size_t threads = 512;\
    const size_t float4_size = (size + 3) / 4;\
    const size_t blocks = (float4_size + threads - 1) / threads;\
    buffer_operation_kernel<op><<<blocks, threads>>>(size, in, out);\
}\

#define BACKPROP(name, op)\
extern "C" void name(const size_t size, const float* input, const float* output_grad, float* input_grad)\
{\
    const size_t threads = 512;\
    const size_t float4_size = (size + 3) / 4;\
    const size_t blocks = (float4_size + threads - 1) / threads;\
    buffer_backprop_kernel<op><<<blocks, threads>>>(size, input, output_grad, input_grad);\
}\

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
__global__ void buffer_backprop_kernel(const int32_t size, const float* input, const float* output_grad, float* input_grad)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

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
        for (int32_t i = 0; i < size - 4 * tid; i++)
        {
            const int32_t j = 4 * tid + i;
            input_grad[j] += op(input[j]) * output_grad[j];
        }
    }
}

ACTIVATE(activateReLU, ReLU)
ACTIVATE(activateCReLU, CReLU)
ACTIVATE(activateSCReLU, SCReLU)
ACTIVATE(activateSqrReLU, SqrReLU)
ACTIVATE(activateSigmoid, sigmoid)

BACKPROP(backpropReLU, primeReLU)
BACKPROP(backpropCReLU, primeCReLU)
BACKPROP(backpropSCReLU, primeSCReLU)
BACKPROP(backpropSqrReLU, primeSqrReLU)
BACKPROP(backpropSigmoid, primeSigmoid)
