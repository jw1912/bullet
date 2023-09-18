/*
Rust constants and directives are passed via macros.
If anyone has a simpler way, please PR it.
*/

constexpr size_t calcBlocks(size_t total, size_t threads)
{
    return (total + threads - 1) / threads;
}

#ifndef HIDDEN
#define HIDDEN 768
#endif

#ifndef INPUT
#define INPUT 32
#endif

#if defined(RELU)
    __device__ float activate(float in) { return in > 0.0F ? in : 0.0F; }
    __device__ float prime(float in) { return in > 0.0F ? 1.0F : 0.0F; }
#elif defined(SCRELU)
    __device__ float activate(float in) { return in < 0.0F ? 0.0F : (in > 1.0F ? 1.0F : (in * in)); }
    __device__ float prime(float in) { return in > 0.0F && in < 1.0F ? 2.0F * sqrt(in) : 0.0F; }
#elif defined(FASTSCRELU)
    constexpr float fastFactor = 255.0F / 256.0F;
    __device__ float activate(float in)
    {
        const float sq = in * in * fastFactor;
        return sq > 1.0F ? 1.0F : sq;
    }
    __device__ float prime(float in) { return fastFactor * (in > 0.0F && in < 1.0F ? 2.0F * sqrt(in) : 0.0F); }
#elif defined(CRELU)
    __device__ float activate(float in) { return in < 0.0F ? 0.0F : (in > 1.0F ? 1.0F : in); }
    __device__ float prime(float in) { return in > 0.0F && in < 1.0F ? 1.0F : 0.0F; }
#else
    __device__ float activate(float in);
    __device__ float prime(float in);
#endif
