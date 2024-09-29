#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(deref_nullptr)]
#![allow(missing_debug_implementations)]
#![allow(improper_ctypes)]
#![allow(unused)]
#![allow(clippy::enum_variant_names)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[link(name = "kernels", kind = "static")]
extern "C" {
    pub fn updateWeights(
        networkSize: usize,
        decay: f32,
        beta1: f32,
        beta2: f32,
        minWeight: f32,
        maxWeight: f32,
        adj: f32,
        rate: f32,
        network: *mut f32,
        momentum: *mut f32,
        velocity: *mut f32,
        gradients: *const f32,
    );

    pub fn sparseLinearForward(
        batchSize: usize,
        maxInputSize: usize,
        outputSize: usize,
        weights: *const f32,
        inputs: *const i32,
        outputs: *mut f32,
    );

    pub fn sparseLinearBackward(
        batchSize: usize,
        maxInputSize: usize,
        outputSize: usize,
        weightsGrad: *mut f32,
        inputs: *const i32,
        errors: *const f32,
        output: *const f32,
    );

    pub fn activateReLU(size: usize, inp: *const f32, out: *mut f32);

    pub fn activateCReLU(size: usize, inp: *const f32, out: *mut f32);

    pub fn activateSCReLU(size: usize, inp: *const f32, out: *mut f32);

    pub fn activateSqrReLU(size: usize, inp: *const f32, out: *mut f32);

    pub fn backpropReLU(size: usize, inp: *const f32, out: *mut f32);

    pub fn backpropCReLU(size: usize, inp: *const f32, out: *mut f32);

    pub fn backpropSCReLU(size: usize, inp: *const f32, out: *mut f32);

    pub fn backpropSqrReLU(size: usize, inp: *const f32, out: *mut f32);

    pub fn sigmoidMPE(bufferSize: usize, outputs: *mut f32, results: *const f32, error: *mut f32, power: f32);

    pub fn splatAdd(batchSize: usize, tensorSize: usize, inp_a: *const f32, inp_b: *const f32, out: *mut f32);

    pub fn activateDual(batchSize: usize, tensorSize: usize, inp: *const f32, out: *mut f32);

    pub fn backpropDual(batchSize: usize, tensorSize: usize, inp: *const f32, out: *mut f32);

    pub fn selectForward(
        batchSize: usize,
        inputSize: usize,
        outputSize: usize,
        buckets: *const u8,
        inp: *const f32,
        out: *mut f32,
    );

    pub fn selectBackprop(
        batchSize: usize,
        inputSize: usize,
        outputSize: usize,
        buckets: *const u8,
        inp: *const f32,
        out: *mut f32,
    );

    pub fn pairwiseMul(batchSize: usize, inputSize: usize, outputSize: usize, input: *const f32, output: *mut f32);

    pub fn backpropPairwiseMul(
        batchSize: usize,
        inputSize: usize,
        outputSize: usize,
        input: *const f32,
        output: *mut f32,
    );
}
