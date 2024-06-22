#include <metal_stdlib>

using namespace metal;

float ReLU  (float x) { return x > 0.0F ? x : 0.0F; }
float CReLU (float x) { return x < 0.0F ? 0.0F : (x > 1.0F ? 1.0F : x); }
float SCReLU(float x) { return x < 0.0F ? 0.0F : (x > 1.0F ? 1.0F : (x * x)); }

kernel void activateReLU(
    constant uint &size [[ buffer(0) ]],
    device float* in  [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    uint i [[thread_position_in_grid]])
{
    if (i >= size) return;
    out[i] = ReLU(in[i]);
}

kernel void activateCReLU(
    constant uint &size [[ buffer(0) ]],
    device float* in  [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    uint i [[thread_position_in_grid]])
{
    if (i >= size) return;
    out[i] = CReLU(in[i]);
}

kernel void activateSCReLU(
    constant uint &size [[ buffer(0) ]],
    device float* in  [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    uint i [[thread_position_in_grid]])
{
    if (i >= size) return;
    out[i] = SCReLU(in[i]);
}

// No of Threads = tensor size
//kernel void activateDual(
//    constant size_t &batchSize [[ buffer(0) ]],
//    device float* inp [[ buffer(2) ]],
//    device float* out [[ buffer(3) ]],
//    uint tid [[thread_position_in_grid]])
//{
//    const float thisInp = inp[tensorSize * blockIdx.y + tid];
//    float* thisOut = out + 2 * tensorSize * blockIdx.y + tid;
//
//    thisOut[0] = CReLU(thisInp);
//    thisOut[tensorSize] = SCReLU(thisInp);
//}

kernel void addTo(
    constant uint &size [[ buffer(0) ]],
    device float* in  [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    uint i [[thread_position_in_grid]])
{
    if (i >= size) return;
    out[i] += in[i];
}
