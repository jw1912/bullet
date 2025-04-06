#ifndef BULLET_CUDA_UTILS
#define BULLET_CUDA_UTILS
#include "util.cu"
#endif

BULLET_KERNEL ScaleAssignKernel(const int size, float* params, const float alpha) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        float4 a = ((float4 *)params)[tid];
        
        a.x *= alpha;
        a.y *= alpha;
        a.z *= alpha;
        a.w *= alpha;

        ((float4 *)params)[tid] = a;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            params[j] *= alpha;
        }
    }
}

BULLET_KERNEL ScaleAddAssignKernel(const int size, const float alpha, float* ap, const float beta, const float* bp) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        float4 a = ((float4 *)ap)[tid];
        const float4 b = ((const float4 *)bp)[tid];
        
        a.x = alpha * a.x + beta * b.x;
        a.y = alpha * a.y + beta * b.y;
        a.z = alpha * a.z + beta * b.z;
        a.w = alpha * a.w + beta * b.w;

        ((float4 *)ap)[tid] = a;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            ap[j] = alpha * ap[j] + beta * bp[j];
        }
    }
}

BULLET_KERNEL ScaleKernel(const int size, const float alpha, const float* inp, float* out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        float4 a = ((float4 *)out)[tid];
        const float4 b = ((const float4 *)inp)[tid];
        
        a.x = alpha * b.x;
        a.y = alpha * b.y;
        a.z = alpha * b.z;
        a.w = alpha * b.w;

        ((float4 *)out)[tid] = a;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            out[j] = alpha * inp[j];
        }
    }
}

BULLET_KERNEL LinearCombKernel(const int size, const float alpha, const float* ap, const float beta, const float* bp, float* cp) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        const float4 a = ((const float4 *)ap)[tid];
        const float4 b = ((const float4 *)bp)[tid];
        float4 c = ((float4 *)cp)[tid];
        
        c.x = alpha * a.x + beta * b.x;
        c.y = alpha * a.y + beta * b.y;
        c.z = alpha * a.z + beta * b.z;
        c.w = alpha * a.w + beta * b.w;

        ((float4 *)cp)[tid] = c;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            cp[j] = alpha * ap[j] + beta * bp[j];
        }
    }
}
