#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(deref_nullptr)]
#![allow(missing_debug_implementations)]
#![allow(improper_ctypes)]
#![allow(unused)]

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
        chunkSize: usize,
        maxInputSize: usize,
        outputSize: usize,
        weights: *const f32,
        biases: *const f32,
        inputs: *const u16,
        outputs: *mut f32,
    );

    pub fn sparseAffineBackward(
        batchSize: usize,
        chunkSize: usize,
        maxInputSize: usize,
        outputSize: usize,
        weightsGrad: *mut f32,
        biasesGrad: *mut f32,
        inputs: *const u16,
        errors: *const f32,
    );

    pub fn activateReLU(
        bufferSize: usize,
        buffer: *mut f32,
    );

    pub fn activateCReLU(
        bufferSize: usize,
        buffer: *mut f32,
    );

    pub fn activateSCReLU(
        bufferSize: usize,
        buffer: *mut f32,
    );

    pub fn backpropReLU(
        bufferSize: usize,
        buffer: *mut f32,
    );

    pub fn backpropCReLU(
        bufferSize: usize,
        buffer: *mut f32,
    );

    pub fn backpropSCReLU(
        bufferSize: usize,
        buffer: *mut f32,
    );

    pub fn sigmoidMSE(
        bufferSize: usize,
        outputs: *mut f32,
        results: *const f32,
        error: *mut f32,
    );
}
