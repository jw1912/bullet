#ifndef STUFF
#define INV_DERIV 1.0F
#define BIAS_ARG
#define DECL_MAXY 1
#define DECL_M 1
#define DECL_NNZ 1
#define BIAS_BACKPROP
#endif

constexpr int MaximumBlocksY = DECL_MAXY;
constexpr int m = DECL_M;
constexpr int nnz = DECL_NNZ;

__device__ float op([[maybe_unused]] float x) {
    return INV_DERIV;
}

extern "C" __global__ void kernel(
    const int k,
    const int* X,
    const float* Y,
    const float* Yg,
    float* Ag
    BIAS_ARG)
{
    const int loc = MaximumBlocksY * blockIdx.z + blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || loc >= k)
        return;

    const int* tX = X + nnz * loc;
    const int offset = m * loc;

    const float tE = op(Y[offset + row]) * Yg[offset + row];

    BIAS_BACKPROP

    for (int i = 0; i < nnz; i++) {
        const int j = tX[i];

        if (j == -1)
            break;

        if (tE != 0.0F)
            atomicAdd(&Ag[j * m + row], tE);
    }
}
