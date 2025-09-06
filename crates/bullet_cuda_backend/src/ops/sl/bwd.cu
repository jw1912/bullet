#ifndef STUFF
#define INV_DERIV 1.0F
#define DECL_MAXY 16384
#define DECL_M 128
#define DECL_NNZ 32
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
    const float* HL,
    const float* OUTg,
    const float* OW,
    float* OWg,
    float* Wg,
    float* Bg)
{
    const int loc = MaximumBlocksY * blockIdx.z + blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || loc >= k)
        return;

    const int* tX = X + nnz * loc;
    const int offset = m * loc;

    const float grd = OUTg[loc];
    const float tHL = HL[m * loc + row];

    atomicAdd(OWg + row, grd * tHL);
    const float tE = op(tHL) * grd * OW[row];

    if (tE != 0.0F)
        atomicAdd(Bg + row, tE);

    for (int i = 0; i < nnz; i++) {
        const int j = tX[i];

        if (j == -1)
            break;

        if (tE != 0.0F)
            atomicAdd(Wg + j * m + row, tE);
    }
}
