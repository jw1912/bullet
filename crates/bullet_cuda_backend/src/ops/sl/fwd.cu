#ifndef STUFF
#define ACT_FN x
#define DECL_MAXY 16384
#define DECL_M 128
#define DECL_NNZ 32
#endif

constexpr int MaximumBlocksY = DECL_MAXY;
constexpr int m = DECL_M;
constexpr int nnz = DECL_NNZ;

__device__ float op([[maybe_unused]] float x) {
    return ACT_FN;
}

extern "C" __global__ void kernel(
    const int k,
    const float4* W,
    const float4* B,
    const float4* OW,
    const int* X,
    float4* HL,
    float* OUT)
{
    const int loc = MaximumBlocksY * blockIdx.x + blockIdx.y;
    const int row = threadIdx.x;

    if (4 * row >= m || loc >= k) return;

    float4 sum = B[row];

    for (int i = 0; i < nnz; i++) {
        const int j = X[nnz * loc + i];

        if (j == -1)
            break;

        const float4 a = W[j * m / 4 + row];

        sum.x += a.x;
        sum.y += a.y;
        sum.z += a.z;
        sum.w += a.w;
    }

    sum.x = op(sum.x);
    sum.y = op(sum.y);
    sum.z = op(sum.z);
    sum.w = op(sum.w);

    HL[m * loc / 4 + row] = sum;

    const float4 tOW = OW[row];
    float partial = sum.x * tOW.x + sum.y * tOW.y + sum.z * tOW.z + sum.w * tOW.w;

    partial += __shfl_down_sync(0xffffffff, partial, 16);
    partial += __shfl_down_sync(0xffffffff, partial, 8);
    partial += __shfl_down_sync(0xffffffff, partial, 4);
    partial += __shfl_down_sync(0xffffffff, partial, 2);
    partial += __shfl_down_sync(0xffffffff, partial, 1);

    if ((row & 31) == 0)
        atomicAdd(OUT + loc, partial);
}
