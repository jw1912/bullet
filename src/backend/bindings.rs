#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(deref_nullptr)]
#![allow(missing_debug_implementations)]
#![allow(improper_ctypes)]
#![allow(unused)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::useless_transmute)]

#[cfg(not(feature = "hip"))]
mod cuda;

#[cfg(not(feature = "hip"))]
pub use cuda::*;

#[cfg(feature = "hip")]
pub use hip_bindings::*;

#[cfg(feature = "hip")]
mod hip_bindings {
    pub use raw_bindings::{
        hipDeviceSynchronize as cudaDeviceSynchronize, hipError_t as cudaError_t, hipFree as cudaFree,
        hipGetLastError as cudaGetLastError, hipMalloc as cudaMalloc, hipMemcpy as cudaMemcpy,
        hipMemcpyKind as cudaMemcpyKind, hipMemset as cudaMemset, hipblasCreate as cublasCreate_v2,
        hipblasDestroy as cublasDestroy_v2, hipblasHandle_t as cublasHandle_t, hipblasOperation_t as cublasOperation_t,
        hipblasSaxpy as cublasSaxpy_v2, hipblasSgeam as cublasSgeam, hipblasSgemm as cublasSgemm_v2,
        hipblasSgemv as cublasSgemv_v2, hipblasStatus_t as cublasStatus_t,
    };

    pub const H2D: cudaMemcpyKind = cudaMemcpyKind::hipMemcpyHostToDevice;
    pub const D2H: cudaMemcpyKind = cudaMemcpyKind::hipMemcpyDeviceToHost;
    pub const D2D: cudaMemcpyKind = cudaMemcpyKind::hipMemcpyDeviceToDevice;
    pub const SUCCESS: cudaError_t = cudaError_t::hipSuccess;
    pub const CUBLAS_SUCCESS: cublasStatus_t = cublasStatus_t::HIPBLAS_STATUS_SUCCESS;
    pub const CUBLAS_OP_N: cublasOperation_t = cublasOperation_t::HIPBLAS_OP_N;
    pub const CUBLAS_OP_T: cublasOperation_t = cublasOperation_t::HIPBLAS_OP_T;

    mod raw_bindings {
        include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
    }
}
