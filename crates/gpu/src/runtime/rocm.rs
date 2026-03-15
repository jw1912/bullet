//! Minimal wrapper around the CUDA runtime

use std::{
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    mem::MaybeUninit,
};

use crate::runtime::bindings::GemmConfig;

use super::bindings::{Dim3, GpuBindings};

use raw::*;

/// Marker for the CUDA runtime
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ROCm;

/// Error type for the CUDA runtime
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ROCmError {
    Driver(String),
    Blas(String),
    Nvrtc(String),
    Message(String),
}

impl From<String> for ROCmError {
    fn from(value: String) -> Self {
        Self::Message(value)
    }
}

type ROCmResult = Result<(), ROCmError>;

#[allow(unsafe_op_in_unsafe_fn)]
impl GpuBindings for ROCm {
    type Err = ROCmError;
    type Dev = c_int;
    type Ptr = *mut c_void;
    type Ctx = c_int;
    type Stream = hipStream;
    type BlasHandle = hipblasHandle;
    type Kernel = hipFunction;
    type Module = hipModule;

    unsafe fn driver_init() -> ROCmResult {
        Ok(())
    }

    unsafe fn device_get(ordinal: c_int) -> Result<c_int, ROCmError> {
        Ok(ordinal)
    }

    unsafe fn context_create(device: c_int) -> Result<c_int, ROCmError> {
        Ok(device)
    }

    unsafe fn context_destroy(_device: c_int) -> Result<(), Self::Err> {
        Ok(())
    }

    unsafe fn context_set(ctx: c_int) -> ROCmResult {
        error::runtime(hipSetDevice(ctx))
    }

    unsafe fn stream_create() -> Result<hipStream, ROCmError> {
        let mut stream = MaybeUninit::uninit();
        error::runtime(hipStreamCreate(stream.as_mut_ptr(), 0))?;
        Ok(stream.assume_init())
    }

    unsafe fn stream_destroy(stream: hipStream) -> ROCmResult {
        error::runtime(hipStreamDestroy(stream))
    }

    unsafe fn stream_sync(stream: hipStream) -> ROCmResult {
        error::runtime(hipStreamSynchronize(stream))
    }

    unsafe fn stream_malloc(stream: hipStream, bytes: usize) -> Result<*mut c_void, ROCmError> {
        let mut dev_ptr = MaybeUninit::uninit();
        error::runtime(hipMallocAsync(dev_ptr.as_mut_ptr(), bytes, stream))?;
        Ok(dev_ptr.assume_init())
    }

    unsafe fn stream_free(stream: hipStream, dev_ptr: *mut c_void) -> ROCmResult {
        error::runtime(hipFreeAsync(dev_ptr, stream))
    }

    unsafe fn stream_memset(stream: hipStream, dev_ptr: *mut c_void, bytes: usize, value: u8) -> Result<(), ROCmError> {
        error::runtime(hipMemsetD8Async(dev_ptr, value, bytes, stream))
    }

    unsafe fn stream_memcpy_d2h(stream: hipStream, dst: *mut c_void, src: *mut c_void, bytes: usize) -> ROCmResult {
        error::runtime(hipMemcpyDtoHAsync(dst, src, bytes, stream))
    }

    unsafe fn stream_memcpy_h2d(stream: hipStream, dst: *mut c_void, src: *const c_void, bytes: usize) -> ROCmResult {
        error::runtime(hipMemcpyHtoDAsync(dst, src, bytes, stream))
    }

    unsafe fn kernel_load(_kernel: hipFunction) -> ROCmResult {
        Ok(())
    }

    unsafe fn kernel_launch(
        func: hipFunction,
        stream: hipStream,
        gdim: Dim3,
        bdim: Dim3,
        args: *mut *mut c_void,
        smem: c_uint,
    ) -> ROCmResult {
        error::runtime(hipModuleLaunchKernel(
            func,
            gdim.x,
            gdim.y,
            gdim.z,
            bdim.x,
            bdim.y,
            bdim.z,
            smem as c_uint,
            stream,
            args,
            std::ptr::null_mut(),
        ))
    }

    unsafe fn module_create(code: *const c_void) -> Result<hipModule, ROCmError> {
        let mut module = MaybeUninit::uninit();
        error::runtime(hipModuleLoadData(module.as_mut_ptr(), code))?;
        Ok(module.assume_init())
    }

    unsafe fn module_destroy(module: hipModule) -> ROCmResult {
        error::runtime(hipModuleUnload(module))
    }

    unsafe fn module_get_kernel(module: hipModule, kernel_name: &CStr) -> Result<hipFunction, ROCmError> {
        let mut func = MaybeUninit::uninit();
        error::runtime(hipModuleGetFunction(func.as_mut_ptr(), module, kernel_name.as_ptr()))?;
        Ok(func.assume_init())
    }

    unsafe fn program_compile(
        source_code: &CStr,
        num_options: c_int,
        options: *const *const c_char,
    ) -> Result<Vec<c_char>, ROCmError> {
        let mut prog: hiprtcProgram = std::ptr::null_mut();
        error::nvrtc(hiprtcCreateProgram(
            &mut prog,
            source_code.as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        ))?;
        error::nvrtc(hiprtcCompileProgram(prog, num_options, options))?;

        let mut size = 0usize;
        error::nvrtc(hiprtcGetCodeSize(prog, &mut size))?;

        let mut ptx = vec![0; size];
        error::nvrtc(hiprtcGetCode(prog, ptx.as_mut_ptr()))?;

        error::nvrtc(hiprtcDestroyProgram(&mut prog))?;

        Ok(ptx)
    }

    unsafe fn blas_create() -> Result<hipblasHandle, ROCmError> {
        let mut handle = MaybeUninit::uninit();
        error::blas(hipblasCreate(handle.as_mut_ptr()))?;
        Ok(handle.assume_init())
    }

    unsafe fn blas_destroy(handle: hipblasHandle) -> ROCmResult {
        error::blas(hipblasDestroy(handle))
    }

    unsafe fn blas_set_stream(handle: hipblasHandle, stream: hipStream) -> ROCmResult {
        error::blas(hipblasSetStream(handle, stream))
    }

    unsafe fn blas_gemm(
        handle: hipblasHandle,
        config: GemmConfig,
        a: *mut c_void,
        b: *mut c_void,
        c: *mut c_void,
    ) -> ROCmResult {
        error::blas(hipblasSgemm(
            handle,
            config.row_mjr_a.into(),
            config.row_mjr_b.into(),
            config.m,
            config.n,
            config.k,
            &config.alpha,
            a as *const f32,
            if config.row_mjr_a { config.k } else { config.m },
            b as *const f32,
            if config.row_mjr_b { config.n } else { config.k },
            &config.beta,
            c as *mut f32,
            config.m,
        ))
    }

    unsafe fn blas_gemm_batched(
        handle: hipblasHandle,
        batch_size: c_int,
        config: GemmConfig,
        a: *mut c_void,
        b: *mut c_void,
        c: *mut c_void,
    ) -> ROCmResult {
        error::blas(hipblasSgemmStridedBatched(
            handle,
            config.row_mjr_a.into(),
            config.row_mjr_b.into(),
            config.m,
            config.n,
            config.k,
            &config.alpha,
            a as *const f32,
            if config.row_mjr_a { config.k } else { config.m },
            (config.m * config.k).into(),
            b as *const f32,
            if config.row_mjr_b { config.n } else { config.k },
            (config.k * config.n).into(),
            &config.beta,
            c as *mut f32,
            config.m,
            (config.m * config.n).into(),
            batch_size,
        ))
    }
}

mod error {
    use std::{ffi::CStr, mem::MaybeUninit};

    use super::{ROCmError, ROCmResult, raw::*};

    pub unsafe fn runtime(value: hipError) -> ROCmResult {
        if value == 0 {
            Ok(())
        } else {
            unsafe {
                let mut name = MaybeUninit::uninit();
                hipGetErrorName(value, name.as_mut_ptr());
                let name = CStr::from_ptr(name.assume_init()).to_str().unwrap();

                let mut desc = MaybeUninit::uninit();
                hipGetErrorString(value, desc.as_mut_ptr());
                let desc = CStr::from_ptr(desc.assume_init()).to_str().unwrap();

                Err(ROCmError::Driver(format!("{name}: {desc}")))
            }
        }
    }

    pub unsafe fn blas(value: hipblasStatus) -> ROCmResult {
        if value == 0 {
            Ok(())
        } else {
            unsafe {
                let desc = hipblasStatusToString(value);
                let desc = CStr::from_ptr(desc).to_str().unwrap();
                Err(ROCmError::Blas(desc.into()))
            }
        }
    }

    pub unsafe fn nvrtc(value: hiprtcResult) -> ROCmResult {
        if value == 0 {
            Ok(())
        } else {
            unsafe {
                let desc = hiprtcGetErrorString(value);
                let desc = CStr::from_ptr(desc).to_str().unwrap();
                Err(ROCmError::Nvrtc(desc.to_string()))
            }
        }
    }
}

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(improper_ctypes)]
#[allow(clippy::enum_variant_names)]
mod raw {
    use std::ffi::{c_char, c_int, c_longlong, c_uchar, c_uint, c_void};

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct Opaque {
        _unused: [u8; 0],
    }

    // CUDA Driver

    pub type hipError = c_int;
    pub type hipStream = *mut Opaque;
    pub type hipFunction = *mut Opaque;
    pub type hipModule = *mut Opaque;

    unsafe extern "C" {
        // Errors
        pub fn hipGetErrorString(error: hipError, pStr: *mut *const c_char) -> hipError;
        pub fn hipGetErrorName(error: hipError, pStr: *mut *const c_char) -> hipError;

        // Device
        pub fn hipSetDevice(device: c_int) -> hipError;
        pub fn hipStreamCreate(stream: *mut hipStream, flags: c_uint) -> hipError;
        pub fn hipStreamDestroy(stream: hipStream) -> hipError;
        pub fn hipStreamSynchronize(stream: hipStream) -> hipError;

        // Memory
        pub fn hipMallocAsync(dptr: *mut *mut c_void, bytesize: usize, hStream: hipStream) -> hipError;
        pub fn hipFreeAsync(dptr: *mut c_void, hStream: hipStream) -> hipError;
        pub fn hipMemsetD8Async(dstDevice: *mut c_void, uc: c_uchar, N: usize, hStream: hipStream) -> hipError;
        pub fn hipMemcpyHtoDAsync(
            dstDevice: *mut c_void,
            srcHost: *const c_void,
            ByteCount: usize,
            hStream: hipStream,
        ) -> hipError;
        pub fn hipMemcpyDtoHAsync(
            dstHost: *mut c_void,
            srcDevice: *mut c_void,
            ByteCount: usize,
            hStream: hipStream,
        ) -> hipError;

        // Kernel
        pub fn hipModuleLaunchKernel(
            f: hipFunction,
            gridDimX: c_uint,
            gridDimY: c_uint,
            gridDimZ: c_uint,
            blockDimX: c_uint,
            blockDimY: c_uint,
            blockDimZ: c_uint,
            sharedMemBytes: c_uint,
            hStream: hipStream,
            kernelParams: *mut *mut c_void,
            extra: *mut *mut c_void,
        ) -> hipError;
        pub fn hipModuleLoadData(module: *mut hipModule, image: *const c_void) -> hipError;
        pub fn hipModuleUnload(hmod: hipModule) -> hipError;
        pub fn hipModuleGetFunction(hfunc: *mut hipFunction, hmod: hipModule, name: *const c_char) -> hipError;
    }

    // cuBLAS

    pub type hipblasHandle = *mut Opaque;
    pub type hipblasStatus = c_int;

    #[allow(unused)]
    #[repr(i32)]
    #[non_exhaustive]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub enum hipblasOperation {
        HIPBLAS_OP_N = 111,
        HIPBLAS_OP_T = 112,
        HIPBLAS_OP_C = 113,
    }

    impl From<bool> for hipblasOperation {
        fn from(value: bool) -> Self {
            if value { Self::HIPBLAS_OP_T } else { Self::HIPBLAS_OP_N }
        }
    }

    unsafe extern "C" {
        pub fn hipblasStatusToString(status: hipblasStatus) -> *const c_char;
        pub fn hipblasCreate(handle: *mut hipblasHandle) -> hipblasStatus;
        pub fn hipblasDestroy(handle: hipblasHandle) -> hipblasStatus;
        pub fn hipblasSetStream(handle: hipblasHandle, streamId: hipStream) -> hipblasStatus;
        pub fn hipblasSgemm(
            handle: hipblasHandle,
            transa: hipblasOperation,
            transb: hipblasOperation,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const f32,
            A: *const f32,
            lda: c_int,
            B: *const f32,
            ldb: c_int,
            beta: *const f32,
            C: *mut f32,
            ldc: c_int,
        ) -> hipblasStatus;
        pub fn hipblasSgemmStridedBatched(
            handle: hipblasHandle,
            transa: hipblasOperation,
            transb: hipblasOperation,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const f32,
            A: *const f32,
            lda: c_int,
            strideA: c_longlong,
            B: *const f32,
            ldb: c_int,
            strideB: c_longlong,
            beta: *const f32,
            C: *mut f32,
            ldc: c_int,
            strideC: c_longlong,
            batchCount: c_int,
        ) -> hipblasStatus;
    }

    // NVRTC

    pub type hiprtcResult = c_uint;
    pub type hiprtcProgram = *mut Opaque;

    unsafe extern "C" {
        pub fn hiprtcGetErrorString(result: hiprtcResult) -> *const c_char;
        pub fn hiprtcCompileProgram(
            prog: hiprtcProgram,
            numOptions: c_int,
            options: *const *const c_char,
        ) -> hiprtcResult;
        pub fn hiprtcCreateProgram(
            prog: *mut hiprtcProgram,
            src: *const c_char,
            name: *const c_char,
            numHeaders: c_int,
            headers: *const *const c_char,
            includeNames: *const *const c_char,
        ) -> hiprtcResult;
        pub fn hiprtcDestroyProgram(prog: *mut hiprtcProgram) -> hiprtcResult;
        pub fn hiprtcGetCode(prog: hiprtcProgram, ptx: *const c_char) -> hiprtcResult;
        pub fn hiprtcGetCodeSize(prog: hiprtcProgram, ptxSizeRet: *mut usize) -> hiprtcResult;
    }
}
