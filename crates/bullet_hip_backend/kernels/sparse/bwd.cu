#include "../util.cu"

#if __CUDA_ARCH__ >= 900
template<OpType op>
__global__ void sparse_affine_backward_kernel_aligned(
    const int32_t stride,
    const int32_t nnz,
    const int32_t m,
    const int32_t k,
    const bool Bb,
    const int32_t* X,
    const float* V,
    const float4* Y,
    const float4* Yg,
    float4* Ag,
    float4* Bg)
{
    const int32_t loc = maximumBlocks * blockIdx.z + blockIdx.y;
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (4 * row >= m || loc >= k)
        return;

    const int32_t* tX = X + nnz * loc;
    const int32_t offset = stride * m * loc / 4;

    const float4 tY = Y[offset + row];
    float4 tE = Yg[offset + row];

    tE.x *= op(tY.x);
    tE.y *= op(tY.y);
    tE.z *= op(tY.z);
    tE.w *= op(tY.w);

    if (Bg != nullptr)
    {   
        const int32_t offset2 = Bb ? m * loc / 4 : 0;
        atomicAdd(Bg + offset2 + row, tE);
    }

    for (int32_t i = 0; i < nnz; i++)
    {
        const int32_t j = tX[i];
        const float val = V == nullptr ? 1.0 : V[nnz * loc + i];

        if (j == -1)
            break;

        float4 ttE = tE;
        ttE.x *= val;
        ttE.y *= val;
        ttE.z *= val;
        ttE.w *= val;

        atomicAdd(Ag + j * m / 4 + row, ttE);
    }
}
#endif

template<OpType op>
__global__ void sparse_affine_backward_kernel(
    const int32_t stride,
    const int32_t nnz,
    const int32_t m,
    const int32_t k,
    const bool Bb,
    const int32_t* X,
    const float* V,
    const float* Y,
    const float* Yg,
    float* Ag,
    float* Bg)
{
    const int32_t loc = maximumBlocks * blockIdx.z + blockIdx.y;
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || loc >= k)
        return;

    const int32_t* tX = X + nnz * loc;
    const int32_t offset = stride * m * loc;

    const float tE = op(Y[offset + row]) * Yg[offset + row];

    if (Bg != nullptr && tE != 0.0F)
    {   
        const int32_t offset2 = Bb ? m * loc : 0;
        atomicAdd(&Bg[offset2 + row], tE);
    }

    for (int32_t i = 0; i < nnz; i++)
    {
        const int32_t j = tX[i];
        const float val = V == nullptr ? 1.0 : V[nnz * loc + i];

        if (j == -1)
            break;

        if (tE != 0.0F)
            atomicAdd(Ag + j * m + row, val * tE);
    }
}

template<OpType op>
void sparse_affine_backward_internal(
    const int32_t stride,
    const int32_t nnz,
    const int32_t m,
    const int32_t k,
    const bool Bb,
    const int32_t* X,
    const float* V,
    const float* Y,
    const float* Yg,
    float* Ag,
    float* Bg)
{
    

#if __CUDA_ARCH__ >= 900
    if ((m % 4) == 0 && m >= 128)
    {
        const int32_t max_threads = 1024;
        const int32_t m4_size = (m + 3) / 4; 
        const int32_t threads = min(m4_size, max_threads);
        const int32_t chunks = (m4_size + threads - 1) / threads;

        int32_t ky = min(k, maximumBlocks);
        int32_t kz = (k + maximumBlocks - 1) / maximumBlocks;
        dim3 grid(chunks, ky, kz);

        sparse_affine_backward_kernel_aligned<op><<<grid, threads, alloc>>>(
            stride,
            nnz,
            m,
            k,
            Bb, 
            X,
            V,
            reinterpret_cast<const float4*>(Y),
            reinterpret_cast<const float4*>(Yg),
            reinterpret_cast<float4*>(Ag),
            reinterpret_cast<float4*>(Bg)
        );
    }
    else
    {
#endif
        const int32_t chunks = (m + 1023) / 1024;
        const int32_t threads = (chunks == 1) ? m : 1024;

        int32_t ky = min(k, maximumBlocks);
        int32_t kz = (k + maximumBlocks - 1) / maximumBlocks;
        dim3 grid(chunks, ky, kz);

        sparse_affine_backward_kernel<op><<<grid, threads>>>(stride, nnz, m, k, Bb, X, V, Y, Yg, Ag, Bg);
#if __CUDA_ARCH__ >= 900
    }
#endif
}

extern "C" void sparse_affine_backward(
    const int32_t activation,
    const size_t stride,
    const size_t nnz,
    const size_t m,
    [[maybe_unused]] const size_t n,
    const size_t k,
    const bool Bb,
    const int32_t* X,
    const float* V,
    const float* Y,
    const float* Yg,
    float* Ag,
    float* Bg)
{
    switch (activation)
    {
        case 0:
            sparse_affine_backward_internal<primeInvIdentity>(stride, nnz, m, k, Bb, X, V, Y, Yg, Ag, Bg);
            break;
        case 1:
            sparse_affine_backward_internal<primeInvReLU>(stride, nnz, m, k, Bb, X, V, Y, Yg, Ag, Bg);
            break;
        case 2:
            sparse_affine_backward_internal<primeInvCReLU>(stride, nnz, m, k, Bb, X, V, Y, Yg, Ag, Bg);
            break;
        case 3:
            sparse_affine_backward_internal<primeInvSCReLU>(stride, nnz, m, k, Bb, X, V, Y, Yg, Ag, Bg);
            break;
        case 4:
            sparse_affine_backward_internal<primeInvSqrReLU>(stride, nnz, m, k, Bb, X, V, Y, Yg, Ag, Bg);
            break;
        case 5:
            sparse_affine_backward_internal<primeInvSigmoid>(stride, nnz, m, k, Bb, X, V, Y, Yg, Ag, Bg);
            break;
        default:
            std::abort();
    }
}
