#ifndef BULLET_CUDA_UTILS
#define BULLET_CUDA_UTILS
#include "util.cu"
#endif

constexpr float Epsilon = 0.00000001F;

BULLET_KERNEL_IMPL adamOp(
    const float beta1,
    const float beta2,
    const float adj,
    const float rate,
    const float decay,
    const float wmin,
    const float wmax,
    const bool denom,
    float* p,
    float* m,
    float* v,
    const float* g)
{
    p[0] *= decay;

    const float grad = adj * g[0];
    m[0] = beta1 * m[0] + (1.0F - beta1) * grad;
    v[0] = beta2 * v[0] + (1.0F - beta2) * grad * grad;

    float val = m[0];
    if (denom) val /= sqrt(v[0]) + Epsilon;
    p[0] -= rate * val;

    p[0] = min(max(p[0], wmin), wmax);
}

BULLET_KERNEL AdamKernel(
    const int size,
    const float beta1,
    const float beta2,
    const float adj,
    const float rate,
    const float decay,
    const float min,
    const float max,
    const bool denom,
    float* network,
    float* momentum,
    float* velocity,
    const float* gradients)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        float4 p = ((float4 *)network)[tid];
        float4 m = ((float4 *)momentum)[tid];
        float4 v = ((float4 *)velocity)[tid];
        const float4 g = ((const float4 *)gradients)[tid];

        adamOp(beta1, beta2, adj, rate, decay, min, max, denom, &p.x, &m.x, &v.x, &g.x);
        adamOp(beta1, beta2, adj, rate, decay, min, max, denom, &p.y, &m.y, &v.y, &g.y);
        adamOp(beta1, beta2, adj, rate, decay, min, max, denom, &p.z, &m.z, &v.z, &g.z);
        adamOp(beta1, beta2, adj, rate, decay, min, max, denom, &p.w, &m.w, &v.w, &g.w);

        ((float4 *)network)[tid] = p;
        ((float4 *)momentum)[tid] = m;
        ((float4 *)velocity)[tid] = v;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            adamOp(beta1, beta2, adj, rate, decay, min, max, denom, &network[j], &momentum[j], &velocity[j], &gradients[j]);
        }
    }
}

BULLET_KERNEL ClipKernel(const int size, float* params, const float min_weight, const float max_weight) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        float4 a = ((float4 *)params)[tid];
        
        a.x = min(max(a.x, min_weight), max_weight);
        a.y = min(max(a.y, min_weight), max_weight);
        a.z = min(max(a.z, min_weight), max_weight);
        a.w = min(max(a.w, min_weight), max_weight);

        ((float4 *)params)[tid] = a;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            params[j] = min(max(params[j], min_weight), max_weight);
        }
    }
}
