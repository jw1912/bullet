use bullet_core::{device::blas, graph::ir::shape::Shape};
use cudarc::cublas::{
    sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
    Gemm, GemmConfig, StridedBatchedConfig,
};

use crate::{CudaBuffer, CudaError};

#[allow(unused)]
impl blas::BlasOperations for CudaBuffer<f32> {
    type BlasError = CudaError;

    fn gemm(&mut self, config: &blas::GemmConfig, a: &Self, b: &Self) -> Result<(), Self::BlasError> {
        let (cfg, _) = convert_config(config);

        unsafe { self.device.blas().gemm(cfg, &a.buf, &b.buf, &mut self.buf).map_err(CudaError::Blas) }
    }

    fn gebmm(
        &mut self,
        config: &blas::GemmConfig,
        batch_size: usize,
        a: &Self,
        b: &Self,
    ) -> Result<(), Self::BlasError> {
        let (gemm, shape_o) = convert_config(config);

        let cfg = StridedBatchedConfig {
            gemm,
            batch_size: batch_size as i32,
            stride_a: config.shape_a.size() as i64,
            stride_b: config.shape_b.size() as i64,
            stride_c: shape_o.size() as i64,
        };

        unsafe { self.device.blas().gemm_strided_batched(cfg, &a.buf, &b.buf, &mut self.buf).map_err(CudaError::Blas) }
    }
}

pub fn convert_config(config: &blas::GemmConfig) -> (GemmConfig<f32>, Shape) {
    let blas::GemmConfig { alpha, beta, shape_a, trans_a, shape_b, trans_b } = *config;
    let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);

    let m = if trans_a { shape_a.cols() } else { shape_a.rows() };
    let n = if trans_b { shape_b.rows() } else { shape_b.cols() };
    let k = if trans_a { shape_a.rows() } else { shape_a.cols() };

    if trans_b {
        assert_eq!(shape_b.cols(), k);
    } else {
        assert_eq!(shape_b.rows(), k);
    }

    assert_eq!(shape_o.rows(), m);
    assert_eq!(shape_o.cols(), n);

    let transa = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
    let transb = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };

    let m = m as i32;
    let n = n as i32;
    let k = k as i32;

    let lda = shape_a.rows() as i32;
    let ldb = shape_b.rows() as i32;
    let ldc = shape_o.rows() as i32;

    let cfg = GemmConfig { transa, transb, m, n, k, alpha, lda, ldb, beta, ldc };

    (cfg, shape_o)
}
