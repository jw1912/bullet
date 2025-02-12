#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use cudarc::{
    cublas::{sys::cublasOperation_t::*, CudaBlas, Gemm, GemmConfig, StridedBatchedConfig},
    driver::{CudaView, CudaViewMut},
};

pub unsafe fn sgemm(
    ctx: &CudaBlas,
    input_a: &CudaView<f32>,
    input_a_rows: usize,
    input_a_cols: usize,
    trans_a: bool,
    input_b: &CudaView<f32>,
    input_b_rows: usize,
    input_b_cols: usize,
    trans_b: bool,
    output: &mut CudaViewMut<f32>,
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

    let transa = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
    let transb = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };

    let m = m as i32;
    let n = n as i32;
    let k = k as i32;

    let lda = input_a_rows as i32;
    let ldb = input_b_rows as i32;
    let ldc = output_rows as i32;

    let cfg = GemmConfig { alpha, beta, transa, transb, m, n, k, lda, ldb, ldc };

    unsafe { CudaBlas::gemm(ctx, cfg, input_a, input_b, output).unwrap() }
}

pub unsafe fn batched_sgemm(
    ctx: &CudaBlas,
    batch_size: usize,
    input_a: &CudaView<f32>,
    input_a_rows: usize,
    input_a_cols: usize,
    trans_a: bool,
    input_b: &CudaView<f32>,
    input_b_rows: usize,
    input_b_cols: usize,
    trans_b: bool,
    output: &mut CudaViewMut<f32>,
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

    let transa = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
    let transb = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };

    let m = m as i32;
    let n = n as i32;
    let k = k as i32;

    let lda = input_a_rows as i32;
    let ldb = input_b_rows as i32;
    let ldc = output_rows as i32;

    let stride_a = (input_a_rows * input_a_cols) as i64;
    let stride_b = (input_b_rows * input_b_cols) as i64;
    let stride_c = (output_rows * output_cols) as i64;

    let gemm = GemmConfig { alpha, beta, transa, transb, m, n, k, lda, ldb, ldc };

    let cfg = StridedBatchedConfig { gemm, batch_size: batch_size as i32, stride_a, stride_b, stride_c };

    unsafe {
        CudaBlas::gemm_strided_batched(ctx, cfg, input_a, input_b, output).unwrap();
    }
}
