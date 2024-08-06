/*
Computes MSE(sigmoid(outputs), results)
*/
#include <metal_stdlib>

using namespace metal;

kernel void sigmoidMPE(
    constant uint &size   [[ buffer(0) ]],
    device float* outputs [[ buffer(1) ]],
    device float* results [[ buffer(2) ]],
    device float* error   [[ buffer(3) ]],
    device float &power   [[ buffer(4) ]],
    uint i [[thread_position_in_grid]])
{
    if (i >= size) return;

    const float sigmoid = 1.0F / (1.0F + exp(-outputs[i]));
    const float diff = sigmoid - results[i];
    const float absd = abs(diff);

    outputs[i] = pow(absd, power - 1.0F) * sigmoid * (1.0F - sigmoid);
    outputs[i] = diff > 0.0F ? outputs[i] : -outputs[i];

    *error = pow(absd, power);
}
