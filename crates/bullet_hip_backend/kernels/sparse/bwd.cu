#include "../util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

template<OpType op>
__global__ void sparse_affine_backward_kernel(
    const int32_t stride,
    const int32_t nnz,
    const int32_t m,
    const bool Bb,
    const int32_t* X,
    const float* Y,
    const float* Yg,
    float* Ag,
    float* Bg)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m)
        return;

    const int32_t* tX = X + nnz * blockIdx.y;
    const int32_t offset = stride * m * blockIdx.y;

    const float tE = op(Y[offset + row]) * Yg[offset + row];

    if (Bg != nullptr)
    {   
        const int32_t offset = Bb ? m * blockIdx.y : 0;
        atomicAdd(&Bg[offset + row], tE);
    }

    for (int32_t i = 0; i < nnz; i++)
    {
        const int32_t j = tX[i];

        if (j == -1)
            break;

        atomicAdd(&Ag[j * m + row], tE);
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
    const float* Y,
    const float* Yg,
    float* Ag,
    float* Bg)
{
    const int32_t chunks = (m + 1023) / 1024;
    const int32_t threads = (chunks == 1) ? m : 1024;
    dim3 grid(chunks, k);

    sparse_affine_backward_kernel<op><<<grid, threads>>>(stride, nnz, m, Bb, X, Y, Yg, Ag, Bg);
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
    const float* Y,
    const float* Yg,
    float* Ag,
    float* Bg)
{
    switch (activation)
    {
        case 0:
            sparse_affine_backward_internal<primeInvIdentity>(stride, nnz, m, k, Bb, X, Y, Yg, Ag, Bg);
            break;
        case 1:
            sparse_affine_backward_internal<primeInvReLU>(stride, nnz, m, k, Bb, X, Y, Yg, Ag, Bg);
            break;
        case 2:
            sparse_affine_backward_internal<primeInvCReLU>(stride, nnz, m, k, Bb, X, Y, Yg, Ag, Bg);
            break;
        case 3:
            sparse_affine_backward_internal<primeInvSCReLU>(stride, nnz, m, k, Bb, X, Y, Yg, Ag, Bg);
            break;
        case 4:
            sparse_affine_backward_internal<primeInvSqrReLU>(stride, nnz, m, k, Bb, X, Y, Yg, Ag, Bg);
            break;
        case 5:
            sparse_affine_backward_internal<primeInvSigmoid>(stride, nnz, m, k, Bb, X, Y, Yg, Ag, Bg);
            break;
        default:
            std::abort();
    }
}
