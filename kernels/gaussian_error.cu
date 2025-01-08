#include "util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

// much of this implementation is with reference to https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html

__global__ void gaussianErrorKernel(
    const size_t bufferSize,
    const float* mean_inputs,
    const float* variance_inputs,
    const float* results,
    float* output) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= bufferSize)
        return;

    const float mean = mean_inputs[i];
    const float variance = variance_inputs[i];
    // pytorch does this for stability, we may as well do it too
    const float safe_variance = fmaxf(variance, 1e-6f);
    const float result = results[i];

    // Compute negative log likelihood
    // -ln(P(x)) = 1/2 * ( ln(max(var, eps)) + (mean - target)^2 / max(var, eps) )
    const float diff = mean - result;
    const float loss = 0.5f * (logf(2.0f * M_PI * safe_variance) + (diff * diff) / safe_variance);

    atomicAdd(output, loss);
}

__global__ void backpropGaussianErrorKernel(
    const size_t bufferSize,
    const float* mean_inputs,
    const float* variance_inputs,
    const float* results,
    const float* output_grad,
    float* mean_input_grads,
    float* variance_input_grads) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= bufferSize)
        return;

    const float mean = mean_inputs[i];
    const float variance = variance_inputs[i];
    // pytorch does this for stability, we may as well do it too
    const float safe_variance = fmaxf(variance, 1e-6f);
    const float result = results[i];
    const float grad = *output_grad;

    // Gradient with respect to mean is
    // dL/dμ = (mean - target) / max(var, eps)
    const float diff = mean - result;
    const float mean_grad = diff / safe_variance;
    atomicAdd(&mean_input_grads[i], mean_grad * grad);

    // Gradient with respect to variance is
    // 1/2 *( 1/var - (mean - target)^2 / var^2 )
    const float diff_sq = diff * diff;
    const float variance_grad = 0.5f * (1.0f / safe_variance - diff_sq / (safe_variance * safe_variance));
    atomicAdd(&variance_input_grads[i], variance_grad * grad);
}

extern "C" void gaussianError(
    const size_t bufferSize,
    const float* inputs,
    const float* results,
    float* output,
    const float power) {
    const size_t numBlocks = (bufferSize + threadsPerBlock - 1) / threadsPerBlock;
    gaussianErrorKernel<<<numBlocks, threadsPerBlock>>>(bufferSize, inputs, results, output, power);
}

extern "C" void backpropGaussianError(
    const size_t bufferSize,
    const float* inputs,
    const float* results,
    const float* output_grad,
    float* input_grads,
    const float power) {
    const size_t numBlocks = (bufferSize + threadsPerBlock - 1) / threadsPerBlock;
    backpropGaussianErrorKernel<<<numBlocks, threadsPerBlock>>>(bufferSize, inputs, results, output_grad, input_grads, power);
}
