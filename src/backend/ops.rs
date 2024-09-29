// Every operation has the same safety criteria, pass valid pointers
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use super::{
    bindings::{self, cublasOperation_t},
    ExecutionContext,
};

use std::ffi::c_int;

pub unsafe fn splat_mul_matrix_vector(
    ctx: &ExecutionContext,
    m: usize,
    n: usize,
    a_ptr: *const f32,
    x_ptr: *const f32,
    y_ptr: *mut f32,
    batch_size: usize,
    increment: bool,
) {
    let alpha = 1.0;
    let beta = f32::from(increment);

    let m = m as c_int;
    let n = n as c_int;
    let batch_size = batch_size as c_int;

    unsafe {
        bindings::cublasSgemm_v2(
            ctx.handle,
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_N,
            n,
            batch_size,
            m,
            &alpha,
            a_ptr,
            n,
            x_ptr,
            m,
            &beta,
            y_ptr,
            n,
        );
    }
}

pub unsafe fn splat_mul_matrixt_vector(
    ctx: &ExecutionContext,
    m: usize,
    n: usize,
    a_ptr: *const f32,
    y_ptr: *const f32,
    x_ptr: *mut f32,
    batch_size: usize,
    increment: bool,
) {
    let alpha = 1.0;
    let beta = f32::from(increment);

    let m = m as c_int;
    let n = n as c_int;
    let batch_size = batch_size as c_int;

    unsafe {
        bindings::cublasSgemm_v2(
            ctx.handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            m,
            batch_size,
            n,
            &alpha,
            a_ptr,
            n,
            y_ptr,
            n,
            &beta,
            x_ptr,
            m,
        );
    }
}

pub unsafe fn reduce_add_mul_vector_vectort(
    ctx: &ExecutionContext,
    m: usize,
    n: usize,
    y_ptr: *const f32,
    x_ptr: *const f32,
    a_ptr: *mut f32,
    batch_size: usize,
    increment: bool
) {
    let alpha = 1.0;
    let beta = f32::from(increment);

    let m = m as c_int;
    let n = n as c_int;
    let batch_size = batch_size as c_int;

    unsafe {
        bindings::cublasSgemm_v2(
            ctx.handle,
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_T,
            n,
            m,
            batch_size,
            &alpha,
            y_ptr,
            n,
            x_ptr,
            m,
            &beta,
            a_ptr,
            n,
        );
    }
}

pub unsafe fn reduce_add(
    ctx: &ExecutionContext,
    batch_size: usize,
    out_size: usize,
    inp: *const f32,
    out: *mut f32,
    increment: bool,
) {
    let alpha = 1.0;
    let beta = f32::from(increment);

    let m = batch_size as c_int;
    let n = out_size as c_int;

    bindings::cublasSgemv_v2(
        ctx.handle,
        cublasOperation_t::CUBLAS_OP_N,
        n,
        m,
        &alpha,
        inp,
        n,
        ctx.ones.ptr(),
        1,
        &beta,
        out,
        1,
    );
}

pub unsafe fn activate_relu(_: &ExecutionContext, size: usize, inp: *const f32, out: *mut f32) {
    bindings::activateReLU(size, inp, out);
}

pub unsafe fn activate_crelu(_: &ExecutionContext, size: usize, inp: *const f32, out: *mut f32) {
    bindings::activateCReLU(size, inp, out);
}

pub unsafe fn activate_screlu(_: &ExecutionContext, size: usize, inp: *const f32, out: *mut f32) {
    bindings::activateSCReLU(size, inp, out);
}

pub unsafe fn activate_sqrrelu(_: &ExecutionContext, size: usize, inp: *const f32, out: *mut f32) {
    bindings::activateSqrReLU(size, inp, out);
}

pub unsafe fn backprop_relu(_: &ExecutionContext, size: usize, inp: *const f32, out: *mut f32) {
    bindings::backpropReLU(size, inp, out);
}

pub unsafe fn backprop_crelu(_: &ExecutionContext, size: usize, inp: *const f32, out: *mut f32) {
    bindings::backpropCReLU(size, inp, out);
}

pub unsafe fn backprop_screlu(_: &ExecutionContext, size: usize, inp: *const f32, out: *mut f32) {
    bindings::backpropSCReLU(size, inp, out);
}

pub unsafe fn backprop_sqrrelu(_: &ExecutionContext, size: usize, inp: *const f32, out: *mut f32) {
    bindings::backpropSqrReLU(size, inp, out);
}

pub unsafe fn sigmoid_mpe(
    _: &ExecutionContext,
    buffer_size: usize,
    outputs: *mut f32,
    results: *const f32,
    error: *mut f32,
    power: f32,
) {
    bindings::sigmoidMPE(buffer_size, outputs, results, error, power);
}

pub unsafe fn sparse_linear_backward(
    _: &ExecutionContext,
    batch_size: usize,
    max_input_size: usize,
    output_size: usize,
    weights_grad: *mut f32,
    inputs: *const i32,
    errors: *const f32,
    output: *const f32,
) {
    bindings::sparseLinearBackward(
        batch_size,
        max_input_size,
        output_size,
        weights_grad,
        inputs,
        errors,
        output,
    );
}

pub unsafe fn sparse_linear_forward(
    _: &ExecutionContext,
    batch_size: usize,
    max_input_size: usize,
    output_size: usize,
    weights: *const f32,
    inputs: *const i32,
    outputs: *mut f32,
) {
    bindings::sparseLinearForward(batch_size, max_input_size, output_size, weights, inputs, outputs);
}

pub unsafe fn splat_add(_: &ExecutionContext, batch_size: usize, tensor_size: usize, inp_a: *const f32, inp_b: *const f32, out: *mut f32) {
    bindings::splatAdd(batch_size, tensor_size, inp_a, inp_b, out);
}

pub unsafe fn update_weights(
    _: &ExecutionContext,
    network_size: usize,
    decay: f32,
    beta1: f32,
    beta2: f32,
    min_weight: f32,
    max_weight: f32,
    adj: f32,
    rate: f32,
    network: *mut f32,
    momentum: *mut f32,
    velocity: *mut f32,
    gradients: *const f32,
) {
    bindings::updateWeights(
        network_size,
        decay,
        beta1,
        beta2,
        min_weight,
        max_weight,
        adj,
        rate,
        network,
        momentum,
        velocity,
        gradients,
    );
}

pub unsafe fn select(
    _: &ExecutionContext,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    buckets: *const u8,
    inp: *const f32,
    out: *mut f32,
) {
    bindings::selectForward(batch_size, input_size, output_size, buckets, inp, out);
}

pub unsafe fn select_backprop(
    _: &ExecutionContext,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    buckets: *const u8,
    inp: *const f32,
    out: *mut f32,
) {
    bindings::selectBackprop(batch_size, input_size, output_size, buckets, inp, out);
}

pub unsafe fn add_to(_: &ExecutionContext, size: usize, inp: *const f32, out: *mut f32) {
    bindings::addTo(size, inp, out);
}

pub unsafe fn pairwise_mul(
    _: &ExecutionContext,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    inputs: *const f32,
    outputs: *mut f32,
) {
    bindings::pairwiseMul(batch_size, input_size, output_size, inputs, outputs);
}

pub unsafe fn backprop_pairwise_mul(
    _: &ExecutionContext,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    inputs: *const f32,
    outputs: *mut f32,
) {
    bindings::backpropPairwiseMul(batch_size, input_size, output_size, inputs, outputs);
}
