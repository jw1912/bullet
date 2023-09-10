#pragma once
#include "cuda.h"
#include "cuda_runtime.h"

__global__ void add(float* A, float* B, float* C)
{
    int i = threadIdx.x;

    C[i] = A[i] + B[i];
}