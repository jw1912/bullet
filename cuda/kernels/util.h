/*
Rust constants and directives are passed via macros.
If anyone has a simpler way, please PR it.
*/

#ifndef UTIL
#define UTIL

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
    __device__ float activate(float in) { return in > 0 ? in : 0; }
    __device__ float prime(float in) { return in > 0 ? 1 : 0; }
#elif defined(SCRELU)
    __device__ float activate(float in) { return in < 0 ? 0 : (in > 1 ? 1 : (in * in)); }
    __device__ float prime(float in) { return in > 0 && in < 1 ? 2 * sqrt(in) : 0; }
#elif defined(FASTSCRELU)
    constexpr float fastFactor = 255.0 / 256.0;
    __device__ float activate(float in)
    {
        const float sq = in * in * fastFactor;
        return sq < 0 ? 0 : (sq > 1 ? 1 : sq);
    }
    __device__ float prime(float in) { return fastFactor * (in > 0 && in < 1 ? 2 * sqrt(in) : 0); }
#elif defined(CRELU)
    __device__ float activate(float in) { return in < 0 ? 0 : (in > 1 ? 1 : in); }
    __device__ float prime(float in) { return in > 0 && in < 1 ? 1 : 0; }
#else
    __device__ float activate(float in);
    __device__ float prime(float in);
#endif

#endif