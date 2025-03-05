#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use std::ffi::c_int;

use crate::backend::bindings::cublasStatus_t;

use super::bindings::{self, CUBLAS_OP_N, CUBLAS_OP_T};

use super::ExecutionContext;

pub unsafe fn sgemm(
    ctx: &ExecutionContext,
    alpha: f32,
    input_a: *const f32,
    input_a_rows: usize,
    input_a_cols: usize,
    trans_a: bool,
    input_b: *const f32,
    input_b_rows: usize,
    input_b_cols: usize,
    trans_b: bool,
    beta: f32,
    output: *mut f32,
    output_rows: usize,
    output_cols: usize,
) -> cublasStatus_t {
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

    unsafe {
        bindings::cublasSgemm_v2(
            ctx.cublas, trans_a, trans_b, m, n, k, &alpha, input_a, lda, input_b, ldb, &beta, output, ldo,
        )
    }
}

pub unsafe fn batched_sgemm(
    ctx: &ExecutionContext,
    batch_size: usize,
    alpha: f32,
    input_a: *const f32,
    input_a_rows: usize,
    input_a_cols: usize,
    trans_a: bool,
    input_b: *const f32,
    input_b_rows: usize,
    input_b_cols: usize,
    trans_b: bool,
    beta: f32,
    output: *mut f32,
    output_rows: usize,
    output_cols: usize,
) -> cublasStatus_t {
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

    unsafe {
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
    }
}

pub unsafe fn copy_strided(
    ctx: &ExecutionContext,
    rows: usize,
    cols: usize,
    input_stride: usize,
    input: *const f32,
    output_stride: usize,
    output: *mut f32,
    increment: bool,
) -> cublasStatus_t {
    let alpha = 1.0;
    let beta = f32::from(increment);

    let m = rows as c_int;
    let n = cols as c_int;

    let lda = input_stride as c_int;
    let ldc = output_stride as c_int;

    let (ldb, bptr) = if increment { (ldc, output) } else { (rows as c_int, std::ptr::null_mut()) };

    unsafe {
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
    }
}
