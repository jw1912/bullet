#ifndef BULLET_CUDA_UTILS
#define BULLET_CUDA_UTILS
#include "util.cu"
#endif

#define SPARSE_MATMUL_FWD_KERNEL(name, op)\
BULLET_KERNEL name(\
    const int stride,\
    const int nnz,\
    const int m,\
    const int k,\
    const float* A,\
    const int* X,\
    float* Y)\
{\
    sparse_affine_kernel<op>(stride, nnz, m, k, false, A, X, nullptr, Y);\
}\

#define SPARSE_AFFINE_FWD_KERNEL(name, op)\
BULLET_KERNEL name(\
    const int stride,\
    const int nnz,\
    const int m,\
    const int k,\
    const bool Bb,\
    const float* A,\
    const int* X,\
    const float* B,\
    float* Y)\
{\
    sparse_affine_kernel<op>(stride, nnz, m, k, Bb, A, X, B, Y);\
}\

#define SPARSE_MATMUL_ALIGNED_FWD_KERNEL(name, op)\
BULLET_KERNEL name(\
    const int stride,\
    const int nnz,\
    const int m,\
    const int k,\
    const float* A,\
    const int* X,\
    float* Y)\
{\
    sparse_affine_aligned_kernel<op>(stride, nnz, m, k, false, A, X, nullptr, Y);\
}\

#define SPARSE_AFFINE_ALIGNED_FWD_KERNEL(name, op)\
BULLET_KERNEL name(\
    const int stride,\
    const int nnz,\
    const int m,\
    const int k,\
    const bool Bb,\
    const float* A,\
    const int* X,\
    const float* B,\
    float* Y)\
{\
    sparse_affine_aligned_kernel<op>(stride, nnz, m, k, Bb, A, X, B, Y);\
}\

template<OpType op>
BULLET_KERNEL_IMPL sparse_affine_kernel(
    const int stride,
    const int nnz,
    const int m,
    const int k,
    const bool Bb,
    const float* A,
    const int* X,
    const float* B,
    float* Y)
{
    const int loc = MaximumBlocksY * blockIdx.z + blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || loc >= k)
        return;

    const int offset = Bb ? m * loc : 0;
    float sum = B == nullptr ? 0.0F : B[offset + row];

    for (int i = 0; i < nnz; i++) {
        const int j = X[nnz * loc + i];

        if (j == -1)
            break;

        sum += A[j * m + row];
    }

    Y[stride * m * loc + row] = op(sum);
}

template<OpType op>
BULLET_KERNEL_IMPL sparse_affine_aligned_kernel(
    const int stride,
    const int nnz,
    const int m,
    const int k,
    const bool Bb,
    const float* A,
    const int* X,
    const float* B,
    float* Y)
{
    extern __shared__ int sX[];
    const int loc = MaximumBlocksY * blockIdx.z + blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (4 * row >= m || loc >= k)
        return;

    if (threadIdx.x < nnz)
    {
        for (int i = threadIdx.x; i < nnz; i += blockDim.x)
        {
            sX[i] = X[nnz * loc + i];
        }
    }

    __syncthreads();

    const int offset = Bb ? m * loc / 4 : 0;
    float4 val = B == nullptr ? make_float4(0.0F, 0.0F, 0.0F, 0.0F) : reinterpret_cast<const float4*>(B)[offset + row];

    for (int i = 0; i < nnz; i++) {
        const int j = sX[i];

        if (j == -1)
            break;

        const float4 a = reinterpret_cast<const float4*>(A)[j * m / 4 + row];

        val.x += a.x;
        val.y += a.y;
        val.z += a.z;
        val.w += a.w;
    }

    val.x = op(val.x);
    val.y = op(val.y);
    val.z = op(val.z);
    val.w = op(val.w);

    reinterpret_cast<float4*>(Y)[stride * m * loc / 4 + row] = val;
}

SPARSE_MATMUL_FWD_KERNEL(SparseMatmulFwd, Identity);
SPARSE_MATMUL_FWD_KERNEL(SparseMatmulFwdRelu, ReLU);
SPARSE_MATMUL_FWD_KERNEL(SparseMatmulFwdCrelu, CReLU);
SPARSE_MATMUL_FWD_KERNEL(SparseMatmulFwdScrelu, SCReLU);
SPARSE_MATMUL_FWD_KERNEL(SparseMatmulFwdSqrRelu, SqrReLU);
SPARSE_MATMUL_FWD_KERNEL(SparseMatmulFwdSigmoid, sigmoid);

SPARSE_AFFINE_FWD_KERNEL(SparseAffineFwd, Identity);
SPARSE_AFFINE_FWD_KERNEL(SparseAffineFwdRelu, ReLU);
SPARSE_AFFINE_FWD_KERNEL(SparseAffineFwdCrelu, CReLU);
SPARSE_AFFINE_FWD_KERNEL(SparseAffineFwdScrelu, SCReLU);
SPARSE_AFFINE_FWD_KERNEL(SparseAffineFwdSqrRelu, SqrReLU);
SPARSE_AFFINE_FWD_KERNEL(SparseAffineFwdSigmoid, sigmoid);

SPARSE_MATMUL_ALIGNED_FWD_KERNEL(SparseMatmulAlignedFwd, Identity);
SPARSE_MATMUL_ALIGNED_FWD_KERNEL(SparseMatmulAlignedFwdRelu, ReLU);
SPARSE_MATMUL_ALIGNED_FWD_KERNEL(SparseMatmulAlignedFwdCrelu, CReLU);
SPARSE_MATMUL_ALIGNED_FWD_KERNEL(SparseMatmulAlignedFwdScrelu, SCReLU);
SPARSE_MATMUL_ALIGNED_FWD_KERNEL(SparseMatmulAlignedFwdSqrRelu, SqrReLU);
SPARSE_MATMUL_ALIGNED_FWD_KERNEL(SparseMatmulAlignedFwdSigmoid, sigmoid);

SPARSE_AFFINE_ALIGNED_FWD_KERNEL(SparseAffineAlignedFwd, Identity);
SPARSE_AFFINE_ALIGNED_FWD_KERNEL(SparseAffineAlignedFwdRelu, ReLU);
SPARSE_AFFINE_ALIGNED_FWD_KERNEL(SparseAffineAlignedFwdCrelu, CReLU);
SPARSE_AFFINE_ALIGNED_FWD_KERNEL(SparseAffineAlignedFwdScrelu, SCReLU);
SPARSE_AFFINE_ALIGNED_FWD_KERNEL(SparseAffineAlignedFwdSqrRelu, SqrReLU);
SPARSE_AFFINE_ALIGNED_FWD_KERNEL(SparseAffineAlignedFwdSigmoid, sigmoid);
