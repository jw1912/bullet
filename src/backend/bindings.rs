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
pub use cuda_bindings::*;

#[cfg(feature = "hip")]
pub use hip_bindings::*;

#[cfg(not(feature = "hip"))]
mod cuda_bindings {
    pub const H2D: cudaMemcpyKind = cudaMemcpyKind::cudaMemcpyHostToDevice;
    pub const D2H: cudaMemcpyKind = cudaMemcpyKind::cudaMemcpyDeviceToHost;
    pub const D2D: cudaMemcpyKind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
    pub const SUCCESS: cudaError_t = cudaError_t::cudaSuccess;
    pub const CUBLAS_SUCCESS: cublasStatus_t = cublasStatus_t::CUBLAS_STATUS_SUCCESS;
    pub const CUBLAS_OP_N: cublasOperation_t = cublasOperation_t::CUBLAS_OP_N;
    pub const CUBLAS_OP_T: cublasOperation_t = cublasOperation_t::CUBLAS_OP_T;

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[cfg(feature = "hip")]
mod hip_bindings {
    pub use raw_bindings::{
        hipMalloc as cudaMalloc,
        hipDeviceSynchronize as cudaDeviceSynchronize,
        hipFree as cudaFree,
        hipMemset as cudaMemset,
        hipMemcpy as cudaMemcpy,
        hipMemcpyKind as cudaMemcpyKind,
        hipError_t as cudaError_t,
        hipGetLastError as cudaGetLastError,
        hipblasOperation_t as cublasOperation_t,
        hipblasStatus_t as cublasStatus_t,
        hipblasSgemm as cublasSgemm_v2,
        hipblasSgeam as cublasSgeam,
        hipblasSaxpy as cublasSaxpy_v2,
        hipblasSgemv as cublasSgemv_v2,
        hipblasCreate as cublasCreate_v2,
        hipblasDestroy as cublasDestroy_v2,
        hipblasHandle_t as cublasHandle_t,
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
