#ifndef BULLET_CUDA_UTILS
#define BULLET_CUDA_UTILS

typedef float(*OpType)(float);
typedef float(*BinaryOpType)(float, float);

constexpr size_t threadsPerBlock = static_cast<size_t>(1024);
constexpr int32_t maximumBlocks = 32768;

__device__ float Identity([[maybe_unused]] float in) { return in; }
__device__ float ReLU(float in) { return in > 0.0F ? in : 0.0F; }
__device__ float CReLU(float in) { return in < 0.0F ? 0.0F : (in > 1.0F ? 1.0F : in); }
__device__ float SCReLU(float in) { return in < 0.0F ? 0.0F : (in > 1.0F ? 1.0F : (in * in)); }
__device__ float SqrReLU(float in) { return in < 0.0F ? 0.0F : (in * in); }
__device__ float sigmoid(float in) { return 1.0F / (1.0F + expf(-in)); }

__device__ float primeIdentity([[maybe_unused]] float in) { return 1.0F; }
__device__ float primeReLU(float in) { return in > 0.0F ? 1.0F : 0.0F; }
__device__ float primeCReLU(float in) { return in > 0.0F && in < 1.0F ? 1.0F : 0.0F; }
__device__ float primeSCReLU(float in) { return in > 0.0F && in < 1.0F ? 2.0F * in : 0.0F; }
__device__ float primeSqrReLU(float in) { return in > 0.0F ? 2.0F * in : 0.0F; }
__device__ float primeSigmoid(float in) {
    const float act = sigmoid(in);
    return act * (1.0F - act);
}

__device__ float primeInvIdentity([[maybe_unused]] float in) { return 1.0F; }
__device__ float primeInvReLU(float in) { return in > 0.0F ? 1.0F : 0.0F; }
__device__ float primeInvCReLU(float in) { return in > 0.0F && in < 1.0F ? 1.0F : 0.0F; }
__device__ float primeInvSCReLU(float in) { return in > 0.0F && in < 1.0F ? 2.0F * sqrtf(in) : 0.0F; }
__device__ float primeInvSqrReLU(float in) { return in > 0.0F ? 2.0F * sqrtf(in) : 0.0F; }
__device__ float primeInvSigmoid(float in) { return in * (1.0F - in); }

#endif