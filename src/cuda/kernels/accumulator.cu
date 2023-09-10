#pragma once
#include <cuda_runtime.h>

__global__ void AddFeature(float* Accumulator, float* NetBucket)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Accumulator[i] += NetBucket[i];
}