#include "../util.cu"

constexpr float Epsilon = 0.00000001F;

__device__ __forceinline__ void adamOp(
    const float beta1,
    const float beta2,
    const float adj,
    const float rate,
    const bool denom,
    float* p,
    float* m,
    float* v,
    const float* g)
{
    const float grad = adj * g[0];
    m[0] = beta1 * m[0] + (1.0F - beta1) * grad;
    v[0] = beta2 * v[0] + (1.0F - beta2) * grad * grad;

    float val = m[0];
    if (denom) val /= sqrt(v[0]) + Epsilon;
    p[0] -= rate * val;
}

__global__ void AdamKernel(
    const int32_t size,
    const float beta1,
    const float beta2,
    const float adj,
    const float rate,
    const bool denom,
    float* network,
    float* momentum,
    float* velocity,
    const float* gradients)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        float4 p = ((float4 *)network)[tid];
        float4 m = ((float4 *)momentum)[tid];
        float4 v = ((float4 *)velocity)[tid];
        const float4 g = ((const float4 *)gradients)[tid];

        adamOp(beta1, beta2, adj, rate, denom, &p.x, &m.x, &v.x, &g.x);
        adamOp(beta1, beta2, adj, rate, denom, &p.y, &m.y, &v.y, &g.y);
        adamOp(beta1, beta2, adj, rate, denom, &p.z, &m.z, &v.z, &g.z);
        adamOp(beta1, beta2, adj, rate, denom, &p.w, &m.w, &v.w, &g.w);

        ((float4 *)network)[tid] = p;
        ((float4 *)momentum)[tid] = m;
        ((float4 *)velocity)[tid] = v;
    }
    else if (4 * tid < size)
    {
        for (int32_t i = 0; i < size - 4 * tid; i++)
        {
            const int32_t j = 4 * tid + i;
            adamOp(beta1, beta2, adj, rate, denom, &network[j], &momentum[j], &velocity[j], &gradients[j]);
        }
    }
}

__global__ void ClipKernel(const size_t size, float* params, const float min_weight, const float max_weight) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    params[i] = min(max(params[i], min_weight), max_weight);
}

__global__ void ScaleKernel(const size_t size, float* params, const float alpha) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    params[i] = alpha * params[i];
}

extern "C" void Adam(
    const size_t size,
    const float beta1,
    const float beta2,
    const float adj,
    const float rate,
    const bool denom,
    float* network,
    float* momentum,
    float* velocity,
    const float* gradients)
{
    const size_t threads = 1024;
    const size_t float4_size = (size + 3) / 4;
    const size_t blocks = (float4_size + threads - 1) / threads;
    AdamKernel<<<blocks, threads>>>(
        size,
        beta1,
        beta2,
        adj,
        rate,
        denom,
        network,
        momentum,
        velocity,
        gradients
    );
}

extern "C" void clip(const size_t size, float* params, const float min_weight, const float max_weight) {
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    ClipKernel<<<numBlocks, threadsPerBlock>>>(size, params, min_weight, max_weight);
}

extern "C" void scale(const size_t size, float* params, const float alpha) {
    const size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    ScaleKernel<<<numBlocks, threadsPerBlock>>>(size, params, alpha);
}
