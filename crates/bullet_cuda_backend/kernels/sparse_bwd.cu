#ifndef BULLET_CUDA_UTILS
#define BULLET_CUDA_UTILS
#include "util.cu"
#endif

#define SPARSE_MATMUL_BWD_KERNEL(name, op)\
BULLET_KERNEL name(\
    const int stride,\
    const int nnz,\
    const int m,\
    const int k,\
    const int* X,\
    const float* Y,\
    const float* Yg,\
    float* Ag)\
{\
    sparse_affine_backward_kernel<op>(stride, nnz, m, k, false, X, Y, Yg, Ag, nullptr);\
}\

#define SPARSE_AFFINE_BWD_KERNEL(name, op)\
BULLET_KERNEL name(\
    const int stride,\
    const int nnz,\
    const int m,\
    const int k,\
    const bool Bb,\
    const int* X,\
    const float* Y,\
    const float* Yg,\
    float* Ag,\
    float* Bg)\
{\
    sparse_affine_backward_kernel<op>(stride, nnz, m, k, Bb, X, Y, Yg, Ag, Bg);\
}\

template<OpType op>
BULLET_KERNEL_IMPL sparse_affine_backward_kernel(
    const int stride,
    const int nnz,
    const int m,
    const int k,
    const bool Bb,
    const int* X,
    const float* Y,
    const float* Yg,
    float* Ag,
    float* Bg)
{
    const int loc = MaximumBlocksY * blockIdx.z + blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || loc >= k)
        return;

    const int* tX = X + nnz * loc;
    const int offset = stride * m * loc;

    const float tE = op(Y[offset + row]) * Yg[offset + row];

    if (Bg != nullptr && tE != 0.0F)
    {   
        const int offset2 = Bb ? m * loc : 0;
        atomicAdd(&Bg[offset2 + row], tE);
    }

    for (int i = 0; i < nnz; i++)
    {
        const int j = tX[i];

        if (j == -1)
            break;

        if (tE != 0.0F)
            atomicAdd(&Ag[j * m + row], tE);
    }
}

SPARSE_MATMUL_BWD_KERNEL(SparseMatmulBwd, primeInvIdentity);
SPARSE_MATMUL_BWD_KERNEL(SparseMatmulBwdRelu, primeInvReLU);
SPARSE_MATMUL_BWD_KERNEL(SparseMatmulBwdCrelu, primeInvCReLU);
SPARSE_MATMUL_BWD_KERNEL(SparseMatmulBwdScrelu, primeInvSCReLU);
SPARSE_MATMUL_BWD_KERNEL(SparseMatmulBwdSqrRelu, primeInvSqrReLU);
SPARSE_MATMUL_BWD_KERNEL(SparseMatmulBwdSigmoid, primeInvSigmoid);

SPARSE_AFFINE_BWD_KERNEL(SparseAffineBwd, primeInvIdentity);
SPARSE_AFFINE_BWD_KERNEL(SparseAffineBwdRelu, primeInvReLU);
SPARSE_AFFINE_BWD_KERNEL(SparseAffineBwdCrelu, primeInvCReLU);
SPARSE_AFFINE_BWD_KERNEL(SparseAffineBwdScrelu, primeInvSCReLU);
SPARSE_AFFINE_BWD_KERNEL(SparseAffineBwdSqrRelu, primeInvSqrReLU);
SPARSE_AFFINE_BWD_KERNEL(SparseAffineBwdSigmoid, primeInvSigmoid);
