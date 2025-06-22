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
mod hip;

#[cfg(feature = "hip")]
pub use hip::{
    hipDeviceSynchronize as cudaDeviceSynchronize, hipError_t as cudaError_t, hipFree as cudaFree,
    hipGetLastError as cudaGetLastError, hipMalloc as cudaMalloc, hipMemcpy as cudaMemcpy,
    hipMemcpyAsync as cudaMemcpyAsync, hipMemcpyKind as cudaMemcpyKind, hipMemset as cudaMemset,
    hipStreamCreateWithFlags as cudaStreamCreateWithFlags, hipStreamDestroy as cudaStreamDestroy,
    hipStream_t as cudaStream_t, hipblasCreate as cublasCreate_v2, hipblasDestroy as cublasDestroy_v2,
    hipblasHandle_t as cublasHandle_t, hipblasOperation_t as cublasOperation_t, hipblasSaxpy as cublasSaxpy_v2,
    hipblasSgeam as cublasSgeam, hipblasSgemm as cublasSgemm_v2,
    hipblasSgemmStridedBatched as cublasSgemmStridedBatched, hipblasSger as cublasSger_v2,
    hipblasStatus_t as cublasStatus_t, CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_SUCCESS, D2D, D2H, H2D, SUCCESS,
};
