#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use super::{
    bindings::{self, hipblasOperation_t, hipblasStatus_t},
    DeviceHandles,
};
use crate::loader::Feat;

use std::ffi::c_int;

fn check_status(status: hipblasStatus_t) {
    if status != hipblasStatus_t::HIPBLAS_STATUS_SUCCESS {
        eprintln!("HIPBLAS operation failed with status: {:?}", status);
    }
}

pub unsafe fn splat_mul_matrix_vector(
    handle: &DeviceHandles,
    m: usize,
    n: usize,
    a_ptr: *const f32,
    x_ptr: *const f32,
    y_ptr: *mut f32,
    batch_size: usize,
) {
    let alpha = 1.0;
    let beta = 0.0;

    let m = m as c_int;
    let n = n as c_int;
    let batch_size = batch_size as c_int;

    let status = bindings::hipblasSgemm(
        **handle,
        hipblasOperation_t::HIPBLAS_OP_N,
        hipblasOperation_t::HIPBLAS_OP_N,
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
    check_status(status);
}

pub unsafe fn splat_mul_matrixt_vector(
    handle: &DeviceHandles,
    m: usize,
    n: usize,
    a_ptr: *const f32,
    y_ptr: *const f32,
    x_ptr: *mut f32,
    batch_size: usize,
) {
    let alpha = 1.0;
    let beta = 0.0;

    let m = m as c_int;
    let n = n as c_int;
    let batch_size = batch_size as c_int;

    let status = bindings::hipblasSgemm(
        **handle,
        hipblasOperation_t::HIPBLAS_OP_T,
        hipblasOperation_t::HIPBLAS_OP_N,
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
    check_status(status);
}

pub unsafe fn reduce_add_mul_vector_vectort(
    handle: &DeviceHandles,
    m: usize,
    n: usize,
    y_ptr: *const f32,
    x_ptr: *const f32,
    a_ptr: *mut f32,
    batch_size: usize,
) {
    let alpha = 1.0;
    let beta = 0.0;

    let m = m as c_int;
    let n = n as c_int;
    let batch_size = batch_size as c_int;

    let status = bindings::hipblasSgemm(
        **handle,
        hipblasOperation_t::HIPBLAS_OP_N,
        hipblasOperation_t::HIPBLAS_OP_T,
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
    check_status(status);
}

pub unsafe fn reduce_add(
    handle: &DeviceHandles,
    ones: *const f32,
    batch_size: usize,
    out_size: usize,
    inp: *const f32,
    out: *mut f32,
) {
    let alpha = 1.0;
    let beta = 0.0;

    let m = batch_size as c_int;
    let n = out_size as c_int;

    let status = bindings::hipblasSgemv(
        **handle, 
        hipblasOperation_t::HIPBLAS_OP_N, 
        n, 
        m, 
        &alpha, 
        inp, 
        n, 
        ones, 
        1, 
        &beta, 
        out, 
        1
    );
    check_status(status);
}

pub unsafe fn activate_relu(_: &DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    bindings::activateReLU(size, inp, out);
}

pub unsafe fn activate_crelu(_: &DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    bindings::activateCReLU(size, inp, out);
}

pub unsafe fn activate_screlu(_: &DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    bindings::activateSCReLU(size, inp, out);
}

pub unsafe fn backprop_relu(_: &DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    bindings::backpropReLU(size, inp, out);
}

pub unsafe fn backprop_crelu(_: &DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    bindings::backpropCReLU(size, inp, out);
}

pub unsafe fn backprop_screlu(_: &DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    bindings::backpropSCReLU(size, inp, out);
}

pub unsafe fn sigmoid_mpe(
    _: &DeviceHandles,
    buffer_size: usize,
    outputs: *mut f32,
    results: *const f32,
    error: *mut f32,
    power: f32,
) {
    bindings::sigmoidMPE(buffer_size, outputs, results, error, power);
}

pub unsafe fn sparse_affine_backward(
    _: &DeviceHandles,
    batch_size: usize,
    max_input_size: usize,
    _: usize,
    output_size: usize,
    weights_grad: *mut f32,
    biases_grad: *mut f32,
    inputs: *const Feat,
    errors: *const f32,
    output: *const f32,
    ft_reg: f32,
) {
    bindings::sparseAffineBackward(
        batch_size,
        max_input_size,
        output_size,
        weights_grad,
        biases_grad,
        inputs,
        errors,
        output,
        ft_reg,
    );
}

pub unsafe fn sparse_affine_forward(
    _: &DeviceHandles,
    batch_size: usize,
    max_input_size: usize,
    output_size: usize,
    weights: *const f32,
    biases: *const f32,
    inputs: *const Feat,
    outputs: *mut f32,
) {
    bindings::sparseAffineForward(batch_size, max_input_size, output_size, weights, biases, inputs, outputs);
}

pub unsafe fn single_sparse_affine_backward(
    _: &DeviceHandles,
    batch_size: usize,
    max_input_size: usize,
    _: usize,
    output_size: usize,
    weights_grad: *mut f32,
    biases_grad: *mut f32,
    inputs: *const Feat,
    errors: *const f32,
    output: *const f32,
    ft_reg: f32,
) {
    bindings::singleSparseAffineBackward(
        batch_size,
        max_input_size,
        output_size,
        weights_grad,
        biases_grad,
        inputs,
        errors,
        output,
        ft_reg,
    );
}

pub unsafe fn single_sparse_affine_forward(
    _: &DeviceHandles,
    batch_size: usize,
    max_input_size: usize,
    output_size: usize,
    weights: *const f32,
    biases: *const f32,
    inputs: *const Feat,
    outputs: *mut f32,
) {
    bindings::singleSparseAffineForward(batch_size, max_input_size, output_size, weights, biases, inputs, outputs);
}

pub unsafe fn splat_add(_: &DeviceHandles, batch_size: usize, tensor_size: usize, inp: *const f32, out: *mut f32) {
    bindings::splatAdd(batch_size, tensor_size, inp, out);
}

pub unsafe fn update_weights(
    _: &DeviceHandles,
    network_size: usize,
    decay: f32,
    adj: f32,
    rate: f32,
    network: *mut f32,
    momentum: *mut f32,
    velocity: *mut f32,
    gradients: *const f32,
) {
    bindings::updateWeights(network_size, decay, adj, rate, network, momentum, velocity, gradients);
}

pub unsafe fn select(
    _: &DeviceHandles,
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
    _: &DeviceHandles,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    buckets: *const u8,
    inp: *const f32,
    out: *mut f32,
) {
    bindings::selectBackprop(batch_size, input_size, output_size, buckets, inp, out);
}

pub unsafe fn add_to(_: &DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    bindings::addTo(size, inp, out);
}
