#include "../util.cu"

__global__ void powerErrorKernel(
    const size_t bufferSize,
    const float* inputs,
    const float* results,
    float* output,
    const float power)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= bufferSize)
        return;

    output[i] = powf(abs(inputs[i] - results[i]), power);
}

__global__ void backpropPowerErrorKernel(
    const size_t bufferSize,
    const float* inputs,
    const float* results,
    const float* output_grad,
    float* input_grads,
    const float power)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= bufferSize)
        return;

    const float diff = inputs[i] - results[i];
    const float grad = power * powf(abs(diff), power - 1.0F) * output_grad[i];
    input_grads[i] += diff > 0.0F ? grad : -grad;
}

extern "C" void powerError(
    const size_t bufferSize,
    const float* inputs,
    const float* results,
    float* output,
    const float power)
{
    const size_t numBlocks = (bufferSize + threadsPerBlock - 1) / threadsPerBlock;
    powerErrorKernel<<<numBlocks, threadsPerBlock>>>(bufferSize, inputs, results, output, power);
}

extern "C" void backpropPowerError(
    const size_t bufferSize,
    const float* inputs,
    const float* results,
    const float* output_grad,
    float* input_grads,
    const float power)
{
    const size_t numBlocks = (bufferSize + threadsPerBlock - 1) / threadsPerBlock;
    backpropPowerErrorKernel<<<numBlocks, threadsPerBlock>>>(bufferSize, inputs, results, output_grad, input_grads, power);
}
