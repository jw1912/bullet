#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use std::ffi::c_int;

use crate::tensor::backend::{
    bindings::{self, CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_SUCCESS},
    Buffer,
};

use super::ExecutionContext;

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

    let trans_a = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
    let trans_b = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };

    let m = m as c_int;
    let n = n as c_int;
    let k = k as c_int;

    let lda = input_a_rows as c_int;
    let ldb = input_b_rows as c_int;
    let ldo = output_rows as c_int;

    let status = unsafe {
        bindings::cublasSgemm_v2(
            ctx.cublas, trans_a, trans_b, m, n, k, &alpha, input_a, lda, input_b, ldb, &beta, output, ldo,
        )
    };

    assert_eq!(status, CUBLAS_SUCCESS, "cuBLAS sgemm failed!");
}

pub unsafe fn batched_sgemm(
    ctx: &mut ExecutionContext,
    batch_size: usize,
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

    let trans_a = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
    let trans_b = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };

    let m = m as c_int;
    let n = n as c_int;
    let k = k as c_int;

    let lda = input_a_rows as c_int;
    let ldb = input_b_rows as c_int;
    let ldo = output_rows as c_int;

    let stride_a = (input_a_rows * input_a_cols) as i64;
    let stride_b = (input_b_rows * input_b_cols) as i64;
    let stride_o = (output_rows * output_cols) as i64;

    let status = unsafe {
        bindings::cublasSgemmStridedBatched(
            ctx.cublas,
            trans_a,
            trans_b,
            m,
            n,
            k,
            &alpha,
            input_a,
            lda,
            stride_a,
            input_b,
            ldb,
            stride_b,
            &beta,
            output,
            ldo,
            stride_o,
            batch_size as i32,
        )
    };

    assert_eq!(status, CUBLAS_SUCCESS, "cuBLAS sgemm failed!");
}

/// If `input_a = None` then it takes `input_a = output` (in-place operation).
pub unsafe fn linear_comb_matrices(
    ctx: &mut ExecutionContext,
    rows: usize,
    cols: usize,
    alpha: f32,
    input_a: Option<*const f32>,
    beta: f32,
    input_b: *const f32,
    output: *mut f32,
) {
    let m = rows as c_int;
    let n = cols as c_int;

    let lda = rows as c_int;
    let ldb = rows as c_int;
    let ldc = rows as c_int;

    let status = unsafe {
        bindings::cublasSgeam(
            ctx.cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m,
            n,
            &alpha,
            input_a.unwrap_or(output),
            lda,
            &beta,
            input_b,
            ldb,
            output,
            ldc,
        )
    };

    assert_eq!(status, CUBLAS_SUCCESS, "cuBLAS Sgemm failed!");
}

pub unsafe fn reduce_add_cols(
    ctx: &mut ExecutionContext,
    rows: usize,
    cols: usize,
    input: *const f32,
    output: *mut f32,
    alpha: f32,
    increment: bool,
) {
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
            ctx.cublas,
            CUBLAS_OP_N,
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

    assert_eq!(status, CUBLAS_SUCCESS, "cuBLAS Sgemv failed!");
}

pub unsafe fn copy_strided(
    ctx: &mut ExecutionContext,
    rows: usize,
    cols: usize,
    input_stride: usize,
    input: *const f32,
    output_stride: usize,
    output: *mut f32,
    increment: bool,
) {
    let alpha = 1.0;
    let beta = f32::from(increment);

    let m = rows as c_int;
    let n = cols as c_int;

    let lda = input_stride as c_int;
    let ldc = output_stride as c_int;

    let (ldb, bptr) = if increment { (ldc, output) } else { (rows as c_int, std::ptr::null_mut()) };

    let status = unsafe {
        bindings::cublasSgeam(
            ctx.cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m,
            n,
            &alpha,
            input,
            lda,
            &beta,
            bptr,
            ldb,
            output,
            ldc,
        )
    };

    assert_eq!(status, CUBLAS_SUCCESS, "cuBLAS Sgemm failed!");
}

pub unsafe fn add_vector_to_matrix_columns(
    ctx: &mut ExecutionContext,
    rows: usize,
    cols: usize,
    alpha: f32,
    vector: *const f32,
    matrix: *mut f32,
) {
    let m = rows as c_int;
    let n = cols as c_int;

    let lda = rows as c_int;
    let inc = 1;

    if cols > ctx.ones.size() {
        ctx.ones = Buffer::new(cols);
        ctx.ones.load_from_slice(&vec![1.0; cols]);
    }

    let status =
        unsafe { bindings::cublasSger_v2(ctx.cublas, m, n, &alpha, vector, inc, ctx.ones.ptr(), inc, matrix, lda) };

    assert_eq!(status, CUBLAS_SUCCESS, "cuBLAS Sger failed!");
}
