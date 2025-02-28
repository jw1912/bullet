#include "../util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

template<OpType op>
__global__ void sparse_affine_kernel(
    const int32_t stride,
    const int32_t nnz,
    const int32_t m,
    const float* A,
    const int32_t* X,
    const float* B,
    float* Y)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m)
        return;

    float sum = B == nullptr ? 0.0F : B[row];

    for (int32_t i = 0; i < nnz; i++) {
        const int32_t j = X[nnz * blockIdx.y + i];

        if (j == -1)
            break;

        sum += A[j * m + row];
    }

    Y[stride * m * blockIdx.y + row] = op(sum);
}

template<OpType op>
__global__ void sparse_affine_aligned_kernel(
    const int32_t stride,
    const int32_t nnz,
    const int32_t m,
    const float4* A,
    const int32_t* X,
    const float4* B,
    float4* Y)
{
    extern __shared__ int32_t sX[];
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (4 * row >= m)
        return;

    if (threadIdx.x < nnz)
    {
        for (int32_t i = threadIdx.x; i < nnz; i += blockDim.x)
        {
            sX[i] = X[nnz * blockIdx.y + i];
        }
    }

    __syncthreads();

    float4 val = B == nullptr ? make_float4(0.0F, 0.0F, 0.0F, 0.0F) : B[row];

    for (size_t i = 0; i < nnz; i++) {
        const int32_t j = sX[i];

        if (j == -1)
            break;

        const float4 a = A[j * m / 4 + row];

        val.x += a.x;
        val.y += a.y;
        val.z += a.z;
        val.w += a.w;
    }

    val.x = op(val.x);
    val.y = op(val.y);
    val.z = op(val.z);
    val.w = op(val.w);

    Y[stride * m * blockIdx.y / 4 + row] = val;
}

template<OpType op>
void sparse_affine_internal(
    const int32_t stride,
    const int32_t nnz,
    const int32_t m,
    const int32_t k,
    const float* A,
    const int32_t* X,
    const float* B,
    float* Y)
{
    const int32_t max_threads = 1024;
    const int32_t alloc = nnz * sizeof(int32_t);

    if ((m % 4) == 0 && m >= 128)
    {
        const int32_t m4_size = (m + 3) / 4; 
        const int32_t threads = min(m4_size, max_threads);
        const int32_t chunks = (m4_size + threads - 1) / threads;
        dim3 grid(chunks, k);

        sparse_affine_aligned_kernel<op><<<grid, threads, alloc>>>(
            stride,
            nnz,
            m,
            reinterpret_cast<const float4*>(A),
            X,
            reinterpret_cast<const float4*>(B),
            reinterpret_cast<float4*>(Y)
        );
    }
    else
    {
        const int32_t threads = min(m, max_threads);
        const int32_t chunks = (m + threads - 1) / threads;
        dim3 grid(chunks, k);

        sparse_affine_kernel<op><<<grid, threads>>>(stride, nnz, m, A, X, B, Y);
    }
}

extern "C" void sparse_affine(
    const int32_t activation,
    const size_t stride,
    const size_t nnz,
    const size_t m,
    [[maybe_unused]] const size_t n,
    const size_t k,
    const float* A,
    const int32_t* X,
    const float* B,
    float* Y)
{
    switch (activation)
    {
        case 0:
            sparse_affine_internal<Identity>(stride, nnz, m, k, A, X, B, Y);
            break;
        case 1:
            sparse_affine_internal<ReLU>(stride, nnz, m, k, A, X, B, Y);
            break;
        case 2:
            sparse_affine_internal<CReLU>(stride, nnz, m, k, A, X, B, Y);
            break;
        case 3:
            sparse_affine_internal<SCReLU>(stride, nnz, m, k, A, X, B, Y);
            break;
        case 4:
            sparse_affine_internal<SqrReLU>(stride, nnz, m, k, A, X, B, Y);
            break;
        case 5:
            sparse_affine_internal<sigmoid>(stride, nnz, m, k, A, X, B, Y);
            break;
        default:
            std::abort();
    }
}
