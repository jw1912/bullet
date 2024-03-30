#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(deref_nullptr)]
#![allow(missing_debug_implementations)]
#![allow(improper_ctypes)]
#![allow(unused)]

use crate::loader::Feat;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[link(name = "kernels", kind = "static")]
extern "C" {
    pub fn updateWeights(
        networkSize: usize,
        decay: f32,
        adj: f32,
        rate: f32,
        network: *mut f32,
        momentum: *mut f32,
        velocity: *mut f32,
        gradients: *const f32,
    );

    pub fn sparseAffineForward(
        batchSize: usize,
        maxInputSize: usize,
        outputSize: usize,
        weights: *const f32,
        biases: *const f32,
        inputs: *const Feat,
        outputs: *mut f32,
    );

    pub fn sparseAffineBackward(
        batchSize: usize,
        maxInputSize: usize,
        outputSize: usize,
        weightsGrad: *mut f32,
        biasesGrad: *mut f32,
        inputs: *const Feat,
        errors: *const f32,
        output: *const f32,
        ft_reg: f32,
    );

    pub fn singleSparseAffineForward(
        batchSize: usize,
        maxInputSize: usize,
        outputSize: usize,
        weights: *const f32,
        biases: *const f32,
        inputs: *const Feat,
        outputs: *mut f32,
    );

    pub fn singleSparseAffineBackward(
        batchSize: usize,
        maxInputSize: usize,
        outputSize: usize,
        weightsGrad: *mut f32,
        biasesGrad: *mut f32,
        inputs: *const Feat,
        errors: *const f32,
        output: *const f32,
        ft_reg: f32,
    );

    pub fn activateReLU(size: usize, inp: *const f32, out: *mut f32);

    pub fn activateCReLU(size: usize, inp: *const f32, out: *mut f32);

    pub fn activateSCReLU(size: usize, inp: *const f32, out: *mut f32);

    pub fn backpropReLU(size: usize, inp: *const f32, out: *mut f32);

    pub fn backpropCReLU(size: usize, inp: *const f32, out: *mut f32);

    pub fn backpropSCReLU(size: usize, inp: *const f32, out: *mut f32);

    pub fn sigmoidMPE(bufferSize: usize, outputs: *mut f32, results: *const f32, error: *mut f32, power: f32);

    pub fn splatAdd(batchSize: usize, tensorSize: usize, inp: *const f32, out: *mut f32);

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
}
