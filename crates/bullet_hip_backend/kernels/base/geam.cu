__global__ void ScaleAssignKernel(const int32_t size, float* params, const float alpha) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

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
        for (int32_t i = 0; i < size - 4 * tid; i++)
        {
            const int32_t j = 4 * tid + i;
            params[j] *= alpha;
        }
    }
}

__global__ void ScaleAddAssignKernel(const int32_t size, const float alpha, float* ap, const float beta, const float* bp) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

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
        for (int32_t i = 0; i < size - 4 * tid; i++)
        {
            const int32_t j = 4 * tid + i;
            ap[j] = alpha * ap[j] + beta * bp[j];
        }
    }
}

__global__ void ScaleKernel(const int32_t size, const float alpha, const float* inp, float* out) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

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
        for (int32_t i = 0; i < size - 4 * tid; i++)
        {
            const int32_t j = 4 * tid + i;
            out[j] = alpha * inp[j];
        }
    }
}

__global__ void LinearCombKernel(const int32_t size, const float alpha, const float* ap, const float beta, const float* bp, float* cp) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

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
        for (int32_t i = 0; i < size - 4 * tid; i++)
        {
            const int32_t j = 4 * tid + i;
            cp[j] = alpha * ap[j] + beta * bp[j];
        }
    }
}

extern "C" void scale_assign(const size_t size, float* params, const float alpha) {
    const size_t threads = 1024;
    const size_t float4_size = (size + 3) / 4;
    const size_t blocks = (float4_size + threads - 1) / threads;
    ScaleAssignKernel<<<blocks, threads>>>(size, params, alpha);
}

extern "C" void scale_add_assign(const size_t size, const float alpha, float* ap, const float beta, const float* bp) {
    const size_t threads = 1024;
    const size_t float4_size = (size + 3) / 4;
    const size_t blocks = (float4_size + threads - 1) / threads;
    ScaleAddAssignKernel<<<blocks, threads>>>(size, alpha, ap, beta, bp);
}

extern "C" void scale(const size_t size, const float alpha, const float* inp, float* out) {
    const size_t threads = 1024;
    const size_t float4_size = (size + 3) / 4;
    const size_t blocks = (float4_size + threads - 1) / threads;
    ScaleKernel<<<blocks, threads>>>(size, alpha, inp, out);
}

extern "C" void linear_comb(const size_t size, const float alpha, const float* ap, const float beta, const float* bp, float* cp) {
    const size_t threads = 1024;
    const size_t float4_size = (size + 3) / 4;
    const size_t blocks = (float4_size + threads - 1) / threads;
    LinearCombKernel<<<blocks, threads>>>(size, alpha, ap, beta, bp, cp);
}