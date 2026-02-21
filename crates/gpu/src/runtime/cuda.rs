//! Minimal wrapper around the CUDA runtime

use std::{
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    mem::MaybeUninit,
};

use super::bindings::{Dim3, GpuBindings};

use raw::*;

/// Marker for the CUDA runtime
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Cuda;

/// Error type for the CUDA runtime
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaError {
    Driver(String),
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
        let mut device = MaybeUninit::uninit();
        error::driver(cuDeviceGet(device.as_mut_ptr(), ordinal))?;
        Ok(device.assume_init())
    }

    unsafe fn context_create(device: CUdevice) -> Result<CUcontext, CudaError> {
        let mut ctx = MaybeUninit::uninit();
        error::driver(cuDevicePrimaryCtxRetain(ctx.as_mut_ptr(), device))?;
        Ok(ctx.assume_init())
    }

    unsafe fn context_destroy(device: CUdevice) -> Result<(), Self::Err> {
        error::driver(cuDevicePrimaryCtxRelease_v2(device))
    }

    unsafe fn context_set(ctx: CUcontext) -> CudaResult {
        let mut curr = MaybeUninit::uninit();
        error::driver(cuCtxGetCurrent(curr.as_mut_ptr()))?;
        let curr = curr.assume_init();

        if curr.is_null() || curr != ctx {
            error::driver(cuCtxSetCurrent(ctx))?;
        }

        Ok(())
    }

    unsafe fn stream_create() -> Result<CUstream, CudaError> {
        let mut stream = MaybeUninit::uninit();
        error::driver(cuStreamCreate(stream.as_mut_ptr(), 0))?;
        Ok(stream.assume_init())
    }

    unsafe fn stream_destroy(stream: CUstream) -> CudaResult {
        error::driver(cuStreamDestroy(stream))
    }

    unsafe fn stream_sync(stream: CUstream) -> CudaResult {
        error::driver(cuStreamSynchronize(stream))
    }

    unsafe fn stream_malloc(stream: CUstream, bytes: usize) -> Result<CUdeviceptr, CudaError> {
        let mut dev_ptr = MaybeUninit::uninit();
        error::driver(cuMemAllocAsync(dev_ptr.as_mut_ptr(), bytes, stream))?;
        Ok(dev_ptr.assume_init())
    }

    unsafe fn stream_free(stream: CUstream, dev_ptr: CUdeviceptr) -> CudaResult {
        error::driver(cuMemFreeAsync(dev_ptr, stream))
    }

    unsafe fn stream_memset(stream: CUstream, dev_ptr: CUdeviceptr, bytes: usize, value: u8) -> Result<(), CudaError> {
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
        let mut module = MaybeUninit::uninit();
        error::driver(cuModuleLoadData(module.as_mut_ptr(), code))?;
        Ok(module.assume_init())
    }

    unsafe fn module_destroy(module: CUmodule) -> CudaResult {
        error::driver(cuModuleUnload(module))
    }

    unsafe fn module_get_kernel(module: CUmodule, kernel_name: &CStr) -> Result<CUfunction, CudaError> {
        let mut func = MaybeUninit::uninit();
        error::driver(cuModuleGetFunction(func.as_mut_ptr(), module, kernel_name.as_ptr()))?;
        Ok(func.assume_init())
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
        error::nvrtc(nvrtcCompileProgram(prog, num_options, options))?;

        let mut size = 0usize;
        error::nvrtc(nvrtcGetPTXSize(prog, &mut size))?;

        let mut ptx = vec![0; size];
        error::nvrtc(nvrtcGetPTX(prog, ptx.as_mut_ptr()))?;

        error::nvrtc(nvrtcDestroyProgram(&mut prog))?;

        Ok(ptx)
    }
}

mod error {
    use std::{ffi::CStr, mem::MaybeUninit};

    use super::{CudaError, CudaResult, raw::*};

    pub unsafe fn driver(value: CUresult) -> CudaResult {
        if value == 0 {
            Ok(())
        } else {
            unsafe {
                let mut name = MaybeUninit::uninit();
                cuGetErrorName(value, name.as_mut_ptr());
                let name = CStr::from_ptr(name.assume_init()).to_str().unwrap();

                let mut desc = MaybeUninit::uninit();
                cuGetErrorString(value, desc.as_mut_ptr());
                let desc = CStr::from_ptr(desc.assume_init()).to_str().unwrap();

                Err(CudaError::Driver(format!("{name}: {desc}")))
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
    use std::ffi::{c_char, c_int, c_uchar, c_uint, c_ulonglong, c_void};

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

    unsafe extern "C" {
        // Errors
        pub fn cuGetErrorString(error: CUresult, pStr: *mut *const c_char) -> CUresult;
        pub fn cuGetErrorName(error: CUresult, pStr: *mut *const c_char) -> CUresult;

        // Device
        pub fn cuInit(flags: c_uint) -> CUresult;
        pub fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
        pub fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult;
        pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
        pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
        pub fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
        pub fn cuStreamCreate(stream: *mut CUstream, flags: c_uint) -> CUresult;
        pub fn cuStreamDestroy(stream: CUstream) -> CUresult;
        pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;

        // Memory
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
