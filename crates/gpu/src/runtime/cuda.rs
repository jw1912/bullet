//! Minimal wrapper around the CUDA runtime

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};

use crate::runtime::bindings::{DeviceProps, GemmConfig};

use super::bindings::{Dim3, GpuBindings};

use raw::*;

/// Marker for the CUDA runtime
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Cuda;

/// Error type for the CUDA runtime
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaError {
    Driver(String),
    Blas(String),
    Nvrtc(String),
    Message(String),
}

impl From<String> for CudaError {
    fn from(value: String) -> Self {
        Self::Message(value)
    }
}

type CudaResult = Result<(), CudaError>;

#[allow(unsafe_op_in_unsafe_fn)]
impl GpuBindings for Cuda {
    type Err = CudaError;
    type Dev = CUdevice;
    type Ptr = CUdeviceptr;
    type Ctx = CUcontext;
    type Stream = CUstream;
    type BlasHandle = cublasHandle;
    type Kernel = CUfunction;
    type Module = CUmodule;

    unsafe fn driver_init() -> CudaResult {
        error::driver(cuInit(0))
    }

    unsafe fn device_get(ordinal: c_int) -> Result<CUdevice, CudaError> {
        let mut device = CUdevice::default();
        error::driver(cuDeviceGet(&mut device, ordinal))?;
        Ok(device)
    }

    unsafe fn device_props(device: CUdevice) -> Result<DeviceProps, CudaError> {
        let mut warp_size = 0;
        error::driver(cuDeviceGetAttribute(&mut warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device))?;
        if warp_size != 32 {
            return Err("Warp size on NVIDIA GPU not 32!?!?".to_string().into());
        }

        let mut bytes = vec![0u8; 256];
        error::driver(cuDeviceGetName(bytes.as_mut_ptr().cast(), 256, device))?;
        let cname = CStr::from_bytes_until_nul(&bytes).unwrap();

        let mut mem_pools = 0;
        error::driver(cuDeviceGetAttribute(&mut mem_pools, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device))?;

        let mut mjr = 0;
        error::driver(cuDeviceGetAttribute(&mut mjr, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))?;

        let mut mnr = 0;
        error::driver(cuDeviceGetAttribute(&mut mnr, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))?;

        Ok(DeviceProps {
            warp_size: Some(32),
            name: cname.to_str().unwrap().to_string(),
            stream_mem_alloc: mem_pools > 0,
            vec_atomics: mjr >= 9,
            arch: Some(format!("sm_{mjr}{mnr}")),
        })
    }

    unsafe fn context_create(device: CUdevice) -> Result<CUcontext, CudaError> {
        let mut ctx = CUcontext::default();
        error::driver(cuDevicePrimaryCtxRetain(&mut ctx, device))?;
        Ok(ctx)
    }

    unsafe fn context_destroy(device: CUdevice) -> CudaResult {
        error::driver(cuDevicePrimaryCtxRelease_v2(device))
    }

    unsafe fn context_set(ctx: CUcontext) -> CudaResult {
        let mut curr = CUcontext::default();
        error::driver(cuCtxGetCurrent(&mut curr))?;
        let curr = curr;

        if curr.is_null() || curr != ctx {
            error::driver(cuCtxSetCurrent(ctx))?;
        }

        Ok(())
    }

    unsafe fn context_sync() -> CudaResult {
        error::driver(cuCtxSynchronize())
    }

    unsafe fn context_malloc(bytes: usize) -> Result<CUdeviceptr, CudaError> {
        let mut dev_ptr = CUdeviceptr::default();
        error::driver(cuMemAlloc_v2(&mut dev_ptr, bytes))?;
        Ok(dev_ptr)
    }

    unsafe fn context_free(dev_ptr: CUdeviceptr) -> CudaResult {
        error::driver(cuMemFree_v2(dev_ptr))
    }

    unsafe fn context_memset(dev_ptr: CUdeviceptr, bytes: usize, value: u8) -> CudaResult {
        error::driver(cuMemsetD8_v2(dev_ptr, value, bytes))
    }

    unsafe fn context_memcpy_d2h(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CudaResult {
        error::driver(cuMemcpyDtoH_v2(dst, src, bytes))
    }

    unsafe fn context_memcpy_h2d(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CudaResult {
        error::driver(cuMemcpyHtoD_v2(dst, src, bytes))
    }

    unsafe fn stream_create() -> Result<CUstream, CudaError> {
        let mut stream = CUstream::default();
        error::driver(cuStreamCreate(&mut stream, 0))?;
        Ok(stream)
    }

    unsafe fn stream_destroy(stream: CUstream) -> CudaResult {
        error::driver(cuStreamDestroy(stream))
    }

    unsafe fn stream_sync(stream: CUstream) -> CudaResult {
        error::driver(cuStreamSynchronize(stream))
    }

    unsafe fn stream_malloc(stream: CUstream, bytes: usize) -> Result<CUdeviceptr, CudaError> {
        let mut dev_ptr = CUdeviceptr::default();
        error::driver(cuMemAllocAsync(&mut dev_ptr, bytes, stream))?;
        Ok(dev_ptr)
    }

    unsafe fn stream_free(stream: CUstream, dev_ptr: CUdeviceptr) -> CudaResult {
        error::driver(cuMemFreeAsync(dev_ptr, stream))
    }

    unsafe fn stream_memset(stream: CUstream, dev_ptr: CUdeviceptr, bytes: usize, value: u8) -> CudaResult {
        error::driver(cuMemsetD8Async(dev_ptr, value, bytes, stream))
    }

    unsafe fn stream_memcpy_d2h(stream: CUstream, dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CudaResult {
        error::driver(cuMemcpyDtoHAsync_v2(dst, src, bytes, stream))
    }

    unsafe fn stream_memcpy_h2d(stream: CUstream, dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CudaResult {
        error::driver(cuMemcpyHtoDAsync_v2(dst, src, bytes, stream))
    }

    unsafe fn kernel_load(kernel: CUfunction) -> CudaResult {
        error::driver(cuFuncLoad(kernel))
    }

    unsafe fn kernel_launch(
        func: CUfunction,
        stream: CUstream,
        gdim: Dim3,
        bdim: Dim3,
        args: *mut *mut c_void,
        smem: c_uint,
    ) -> CudaResult {
        error::driver(cuLaunchKernel(
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

    unsafe fn module_create(code: *const c_void) -> Result<CUmodule, CudaError> {
        let mut module = CUmodule::default();
        error::driver(cuModuleLoadData(&mut module, code))?;
        Ok(module)
    }

    unsafe fn module_destroy(module: CUmodule) -> CudaResult {
        error::driver(cuModuleUnload(module))
    }

    unsafe fn module_get_kernel(module: CUmodule, kernel_name: &CStr) -> Result<CUfunction, CudaError> {
        let mut func = CUfunction::default();
        error::driver(cuModuleGetFunction(&mut func, module, kernel_name.as_ptr()))?;
        Ok(func)
    }

    unsafe fn program_compile(
        source_code: &CStr,
        num_options: c_int,
        options: *const *const c_char,
    ) -> Result<Vec<c_char>, CudaError> {
        let mut prog: nvrtcProgram = std::ptr::null_mut();
        error::nvrtc(nvrtcCreateProgram(
            &mut prog,
            source_code.as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        ))?;

        let code = (|| {
            error::nvrtc(nvrtcCompileProgram(prog, num_options, options))?;

            let mut size = 0usize;
            error::nvrtc(nvrtcGetPTXSize(prog, &mut size))?;

            let mut ptx = vec![0; size];
            error::nvrtc(nvrtcGetPTX(prog, ptx.as_mut_ptr()))?;

            Ok(ptx)
        })();

        error::nvrtc(nvrtcDestroyProgram(&mut prog))?;

        code
    }

    unsafe fn blas_create() -> Result<cublasHandle, CudaError> {
        let mut handle = cublasHandle::default();
        error::blas(cublasCreate_v2(&mut handle))?;
        Ok(handle)
    }

    unsafe fn blas_destroy(handle: cublasHandle) -> CudaResult {
        error::blas(cublasDestroy_v2(handle))
    }

    unsafe fn blas_set_stream(handle: cublasHandle, stream: CUstream) -> CudaResult {
        error::blas(cublasSetStream_v2(handle, stream))
    }

    unsafe fn blas_gemm(
        handle: cublasHandle,
        config: GemmConfig,
        a: CUdeviceptr,
        b: CUdeviceptr,
        c: CUdeviceptr,
    ) -> CudaResult {
        error::blas(cublasSgemm_v2(
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
        handle: cublasHandle,
        batch_size: c_int,
        config: GemmConfig,
        a: CUdeviceptr,
        b: CUdeviceptr,
        c: CUdeviceptr,
    ) -> CudaResult {
        error::blas(cublasSgemmStridedBatched(
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
    use std::ffi::CStr;

    use super::{CudaError, CudaResult, raw::*};

    pub unsafe fn driver(value: CUresult) -> CudaResult {
        if value == 0 {
            Ok(())
        } else {
            unsafe {
                let mut name = std::ptr::null();
                cuGetErrorName(value, &mut name);
                let name = CStr::from_ptr(name).to_str().unwrap();

                let mut desc = std::ptr::null();
                cuGetErrorString(value, &mut desc);
                let desc = CStr::from_ptr(desc).to_str().unwrap();

                Err(CudaError::Driver(format!("{name}: {desc}")))
            }
        }
    }

    pub unsafe fn blas(value: cublasStatus) -> CudaResult {
        if value == 0 {
            Ok(())
        } else {
            unsafe {
                let name = cublasGetStatusName(value);
                let name = CStr::from_ptr(name).to_str().unwrap();
                let desc = cublasGetStatusString(value);
                let desc = CStr::from_ptr(desc).to_str().unwrap();
                Err(CudaError::Blas(format!("{name}: {desc}")))
            }
        }
    }

    pub unsafe fn nvrtc(value: nvrtcResult) -> CudaResult {
        if value == 0 {
            Ok(())
        } else {
            unsafe {
                let desc = nvrtcGetErrorString(value);
                let desc = CStr::from_ptr(desc).to_str().unwrap();
                Err(CudaError::Nvrtc(desc.to_string()))
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
    use std::ffi::{c_char, c_int, c_longlong, c_uchar, c_uint, c_ulonglong, c_void};

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct Opaque {
        _unused: [u8; 0],
    }

    // CUDA Driver

    pub type CUdevice = c_int;
    pub type CUresult = c_int;
    pub type CUcontext = *mut Opaque;
    pub type CUstream = *mut Opaque;
    pub type CUdeviceptr = c_ulonglong;
    pub type CUfunction = *mut Opaque;
    pub type CUmodule = *mut Opaque;
    pub type CUdevice_attribute = c_uint;

    pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE: u32 = 10;
    pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: u32 = 75;
    pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: u32 = 76;
    pub const CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: u32 = 115;

    unsafe extern "C" {
        // Errors
        pub fn cuGetErrorString(error: CUresult, pStr: *mut *const c_char) -> CUresult;
        pub fn cuGetErrorName(error: CUresult, pStr: *mut *const c_char) -> CUresult;

        // Device
        pub fn cuInit(flags: c_uint) -> CUresult;
        pub fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
        pub fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult;
        pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
        pub fn cuDeviceGetAttribute(pi: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult;
        pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
        pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
        pub fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
        pub fn cuCtxSynchronize() -> CUresult;
        pub fn cuStreamCreate(stream: *mut CUstream, flags: c_uint) -> CUresult;
        pub fn cuStreamDestroy(stream: CUstream) -> CUresult;
        pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;

        // Memory
        pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
        pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
        pub fn cuMemsetD8_v2(dstDevice: CUdeviceptr, uc: c_uchar, N: usize) -> CUresult;
        pub fn cuMemcpyHtoD_v2(dstDevice: CUdeviceptr, srcHost: *const c_void, ByteCount: usize) -> CUresult;
        pub fn cuMemcpyDtoH_v2(dstHost: *mut c_void, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult;
        pub fn cuMemAllocAsync(dptr: *mut CUdeviceptr, bytesize: usize, hStream: CUstream) -> CUresult;
        pub fn cuMemFreeAsync(dptr: CUdeviceptr, hStream: CUstream) -> CUresult;
        pub fn cuMemsetD8Async(dstDevice: CUdeviceptr, uc: c_uchar, N: usize, hStream: CUstream) -> CUresult;
        pub fn cuMemcpyHtoDAsync_v2(
            dstDevice: CUdeviceptr,
            srcHost: *const c_void,
            ByteCount: usize,
            hStream: CUstream,
        ) -> CUresult;
        pub fn cuMemcpyDtoHAsync_v2(
            dstHost: *mut c_void,
            srcDevice: CUdeviceptr,
            ByteCount: usize,
            hStream: CUstream,
        ) -> CUresult;

        // Kernel
        pub fn cuFuncLoad(function: CUfunction) -> CUresult;
        pub fn cuLaunchKernel(
            f: CUfunction,
            gridDimX: c_uint,
            gridDimY: c_uint,
            gridDimZ: c_uint,
            blockDimX: c_uint,
            blockDimY: c_uint,
            blockDimZ: c_uint,
            sharedMemBytes: c_uint,
            hStream: CUstream,
            kernelParams: *mut *mut c_void,
            extra: *mut *mut c_void,
        ) -> CUresult;
        pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
        pub fn cuModuleUnload(hmod: CUmodule) -> CUresult;
        pub fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const c_char) -> CUresult;
    }

    // cuBLAS

    pub type cublasHandle = *mut Opaque;
    pub type cublasStatus = c_int;

    #[allow(unused)]
    #[repr(i32)]
    #[non_exhaustive]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub enum cublasOperation {
        CUBLAS_OP_N = 0,
        CUBLAS_OP_T = 1,
        CUBLAS_OP_C = 2,
        CUBLAS_OP_CONJG = 3,
    }

    impl From<bool> for cublasOperation {
        fn from(value: bool) -> Self {
            if value { Self::CUBLAS_OP_T } else { Self::CUBLAS_OP_N }
        }
    }

    unsafe extern "C" {
        pub fn cublasGetStatusName(status: cublasStatus) -> *const c_char;
        pub fn cublasGetStatusString(status: cublasStatus) -> *const c_char;
        pub fn cublasCreate_v2(handle: *mut cublasHandle) -> cublasStatus;
        pub fn cublasDestroy_v2(handle: cublasHandle) -> cublasStatus;
        pub fn cublasSetStream_v2(handle: cublasHandle, streamId: CUstream) -> cublasStatus;
        pub fn cublasSgemm_v2(
            handle: cublasHandle,
            transa: cublasOperation,
            transb: cublasOperation,
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
        ) -> cublasStatus;
        pub fn cublasSgemmStridedBatched(
            handle: cublasHandle,
            transa: cublasOperation,
            transb: cublasOperation,
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
        ) -> cublasStatus;
    }

    // NVRTC

    pub type nvrtcResult = c_uint;
    pub type nvrtcProgram = *mut Opaque;

    unsafe extern "C" {
        pub fn nvrtcGetErrorString(result: nvrtcResult) -> *const c_char;
        pub fn nvrtcCompileProgram(prog: nvrtcProgram, numOptions: c_int, options: *const *const c_char)
        -> nvrtcResult;
        pub fn nvrtcCreateProgram(
            prog: *mut nvrtcProgram,
            src: *const c_char,
            name: *const c_char,
            numHeaders: c_int,
            headers: *const *const c_char,
            includeNames: *const *const c_char,
        ) -> nvrtcResult;
        pub fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;
        pub fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *const c_char) -> nvrtcResult;
        pub fn nvrtcGetPTXSize(prog: nvrtcProgram, ptxSizeRet: *mut usize) -> nvrtcResult;
    }
}
