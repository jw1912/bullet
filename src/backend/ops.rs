// Every operation has the same safety criteria, pass valid pointers
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use super::{
    bindings::{self, cublasOperation_t, cublasStatus_t},
    buffer::Buffer,
    ExecutionContext,
};

use std::ffi::c_int;

pub unsafe fn sgemm(
    ctx: &mut ExecutionContext,
    input_a: *const f32,
    input_a_rows: usize,
    input_a_cols: usize,
    trans_a: bool,
    input_b: *const f32,
    input_b_rows: usize,
    input_b_cols: usize,
    trans_b: bool,
    output: *mut f32,
    output_rows: usize,
    output_cols: usize,
    increment: bool,
) {
    let alpha = 1.0;
    let beta = f32::from(increment);

    let m = if trans_a { input_a_cols } else { input_a_rows };
    let n = if trans_b { input_b_rows } else { input_b_cols };
    let k = if trans_a { input_a_rows } else { input_a_cols };

    if trans_b {
        assert_eq!(input_b_cols, k);
    } else {
        assert_eq!(input_b_rows, k);
    }

    assert_eq!(output_rows, m);
    assert_eq!(output_cols, n);

    let trans_a = if trans_a { cublasOperation_t::CUBLAS_OP_T } else { cublasOperation_t::CUBLAS_OP_N };
    let trans_b = if trans_b { cublasOperation_t::CUBLAS_OP_T } else { cublasOperation_t::CUBLAS_OP_N };

    let m = m as c_int;
    let n = n as c_int;
    let k = k as c_int;

    let lda = input_a_rows as c_int;
    let ldb = input_b_rows as c_int;
    let ldo = output_rows as c_int;

    let status = unsafe {
        bindings::cublasSgemm_v2(
            ctx.handle, trans_a, trans_b, m, n, k, &alpha, input_a, lda, input_b, ldb, &beta, output, ldo,
        )
    };

    assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS, "cuBLAS sgemm failed!");
}

pub unsafe fn add_matrices(
    ctx: &mut ExecutionContext,
    rows: usize,
    cols: usize,
    input_a: *const f32,
    input_b: *const f32,
    output: *mut f32,
) {
    let alpha = 1.0;
    let beta = 1.0;

    let m = rows as c_int;
    let n = cols as c_int;

    let lda = rows as c_int;
    let ldb = rows as c_int;
    let ldc = rows as c_int;

    let status = unsafe {
        bindings::cublasSgeam(
            ctx.handle,
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            &alpha,
            input_a,
            lda,
            &beta,
            input_b,
            ldb,
            output,
            ldc,
        )
    };

    assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS, "cuBLAS sgemm failed!");
}

pub unsafe fn add_matrix_to(ctx: &mut ExecutionContext, rows: usize, cols: usize, input: *const f32, output: *mut f32) {
    let alpha = 1.0;

    let n = (rows * cols) as c_int;

    let incx = 1;
    let incy = 1;

    let status = unsafe { bindings::cublasSaxpy_v2(ctx.handle, n, &alpha, input, incx, output, incy) };

    assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS, "cuBLAS sgemm failed!");
}

pub unsafe fn reduce_add_cols(
    ctx: &mut ExecutionContext,
    rows: usize,
    cols: usize,
    input: *const f32,
    output: *mut f32,
    increment: bool,
) {
    let alpha = 1.0;
    let beta = f32::from(increment);

    let m = rows as c_int;
    let n = cols as c_int;

    let lda = rows as c_int;
    let inc = 1;

    if cols > ctx.ones.size() {
        ctx.ones = Buffer::new(cols);
        ctx.ones.load_from_slice(&vec![1.0; cols]);
    }

    let status = unsafe {
        bindings::cublasSgemv_v2(
            ctx.handle,
            cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            &alpha,
            input,
            lda,
            ctx.ones.ptr(),
            inc,
            &beta,
            output,
            1,
        )
    };

    assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS, "cuBLAS sgemm failed!");
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
    bindings::sparseLinearBackward(batch_size, max_input_size, output_size, weights_grad, inputs, errors, output);
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

pub unsafe fn splat_add(
    _: &ExecutionContext,
    batch_size: usize,
    tensor_size: usize,
    inp_a: *const f32,
    inp_b: *const f32,
    out: *mut f32,
) {
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
