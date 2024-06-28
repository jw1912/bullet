// This file contains kernels for calculating in-place backpropagations for the various activation functions. Given a
// buffer of inputs in and a buffer of outputs out with a 1D thread grid with size(out) threads in it, it calculates
// out[i] = in[i] * σ'(out[i]) for the activation function σ.
#include <metal_stdlib>

using namespace metal;

// Derivatives of the activation functions.
float ddxReLU  (float x) { return x > 0.0F ? 1.0F : 0.0F; }
float ddxCReLU (float x) { return x > 0.0F && x < 1.0F ? 1.0F : 0.0F; }
float ddxSCReLU(float x) { return x > 0.0F && x < 1.0F ? 2.0F * x : 0.0F; }

kernel void backpropReLU(
    constant uint &size [[ buffer(0) ]],
    device float* in  [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    uint i [[thread_position_in_grid]])
{
    if (i >= size) return;
    out[i] = in[i] * ddxReLU(out[i]);
}

kernel void backpropCReLU(
    constant uint &size [[ buffer(0) ]],
    device float* in  [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    uint i [[thread_position_in_grid]])
{
    if (i >= size) return;
    out[i] = in[i] * ddxCReLU(out[i]);
}

kernel void backpropSCReLU(
    constant uint* size [[ buffer(0) ]],
    device float* in  [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    uint i [[thread_position_in_grid]])
{
    if (i >= size) return;
    out[i] = in[i] * ddxSCReLU(out[i]);
}
