use std::ffi::{c_int, c_void};

use super::{Dim3, GpuBindings, MemcpyKind};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ROCm;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ROCmError {
    Runtime(raw::hipError),
    Blas(raw::hipblasStatus),
    Message(String),
}

impl From<String> for ROCmError {
    fn from(value: String) -> Self {
        Self::Message(value)
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ROCmStream(raw::hipStream);

type ROCmResult = Result<(), ROCmError>;

impl GpuBindings for ROCm {
    type E = ROCmError;
    type S = ROCmStream;

    unsafe fn device_init(_device: c_int) -> ROCmResult {
        unsafe { err(raw::hipInit(0)) }
    }

    unsafe fn device_set(device: c_int) -> ROCmResult {
        unsafe { err(raw::hipSetDevice(device)) }
    }

    unsafe fn stream_create(stream: *mut ROCmStream) -> ROCmResult {
        unsafe { err(raw::hipStreamCreate(stream.cast())) }
    }

    unsafe fn stream_destroy(stream: ROCmStream) -> ROCmResult {
        unsafe { err(raw::hipStreamDestroy(stream.0)) }
    }

    unsafe fn stream_sync(stream: ROCmStream) -> ROCmResult {
        unsafe { err(raw::hipStreamSynchronize(stream.0)) }
    }

    unsafe fn stream_malloc(stream: ROCmStream, dev_ptr: *mut *mut c_void, bytes: usize) -> ROCmResult {
        unsafe { err(raw::hipMallocAsync(dev_ptr, bytes, stream.0)) }
    }

    unsafe fn stream_free(stream: ROCmStream, dev_ptr: *mut c_void) -> ROCmResult {
        unsafe { err(raw::hipFreeAsync(dev_ptr, stream.0)) }
    }

    unsafe fn stream_memset(stream: Self::S, dev_ptr: *mut c_void, bytes: usize, value: u8) -> Result<(), Self::E> {
        unsafe { err(raw::hipMemsetAsync(dev_ptr, value as c_int, bytes, stream.0)) }
    }

    unsafe fn stream_memcpy(
        stream: ROCmStream,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: MemcpyKind,
    ) -> ROCmResult {
        let kind = match kind {
            MemcpyKind::HostToHost => raw::hipMemcpyKind::hipMemcpyHostToHost,
            MemcpyKind::DeviceToDevice => raw::hipMemcpyKind::hipMemcpyDeviceToDevice,
            MemcpyKind::DeviceToHost => raw::hipMemcpyKind::hipMemcpyDeviceToHost,
            MemcpyKind::HostToDevice => raw::hipMemcpyKind::hipMemcpyHostToDevice,
            MemcpyKind::Default => raw::hipMemcpyKind::hipMemcpyDefault,
        };

        unsafe { err(raw::hipMemcpyAsync(dst, src, bytes, kind, stream.0)) }
    }

    unsafe fn stream_launch_kernel(
        stream: ROCmStream,
        func: *const c_void,
        gdim: Dim3,
        bdim: Dim3,
        args: *mut *mut c_void,
        smem: usize,
    ) -> ROCmResult {
        let gdim = raw::dim3 { x: gdim.x, y: gdim.y, z: gdim.z };
        let bdim = raw::dim3 { x: bdim.x, y: bdim.y, z: bdim.z };
        unsafe { err(raw::hipLaunchKernel(func, gdim, bdim, args, smem, stream.0)) }
    }
}

fn err(value: raw::hipError) -> ROCmResult {
    if value != raw::hipError::hipSuccess { Err(ROCmError::Runtime(value)) } else { Ok(()) }
}

fn status(value: raw::hipblasStatus) -> ROCmResult {
    if value != raw::hipblasStatus::HIPBLAS_STATUS_SUCCESS {
        Err(ROCmError::Blas(value))
    } else {
        Ok(())
    }
}

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(improper_ctypes)]
#[allow(clippy::enum_variant_names)]
mod raw {
    use std::ffi::{c_int, c_uint, c_void};

    #[rustfmt::skip]
    unsafe extern "C" {
        pub fn hipInit(flags: c_uint) -> hipError;
        pub fn hipSetDevice(device: c_int) -> hipError;
        pub fn hipStreamCreate(stream: *mut hipStream) -> hipError;
        pub fn hipStreamDestroy(stream: hipStream) -> hipError;
        pub fn hipStreamSynchronize(stream: hipStream) -> hipError;
        pub fn hipMallocAsync(devPtr: *mut *mut c_void, size: usize, stream: hipStream) -> hipError;
        pub fn hipFreeAsync(devPtr: *mut c_void, stream: hipStream) -> hipError;
        pub fn hipMemsetAsync (devPtr: *mut c_void, value: c_int, count: usize, stream: hipStream) -> hipError;
        pub fn hipMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: hipMemcpyKind, stream: hipStream) -> hipError;
        pub fn hipLaunchKernel(func: *const c_void, gridDim: dim3, blockDim: dim3, args: *mut *mut c_void, sharedMem: usize, stream: hipStream) -> hipError;
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct HipStreamOpaque([u8; 0]);
    pub type hipStream = *mut HipStreamOpaque;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct HipblasContextOpaque([u8; 0]);
    pub type hipblasHandle = *mut HipblasContextOpaque;

    #[repr(i32)]
    #[non_exhaustive]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub enum hipMemcpyKind {
        hipMemcpyHostToHost = 0,
        hipMemcpyHostToDevice = 1,
        hipMemcpyDeviceToHost = 2,
        hipMemcpyDeviceToDevice = 3,
        hipMemcpyDefault = 4,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct dim3 {
        pub x: u32,
        pub y: u32,
        pub z: u32,
    }

    #[repr(i32)]
    #[non_exhaustive]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub enum hipblasStatus {
        HIPBLAS_STATUS_SUCCESS = 0,
        HIPBLAS_STATUS_NOT_INITIALIZED = 1,
        HIPBLAS_STATUS_ALLOC_FAILED = 2,
        HIPBLAS_STATUS_INVALID_VALUE = 3,
        HIPBLAS_STATUS_MAPPING_ERROR = 4,
        HIPBLAS_STATUS_EXECUTION_FAILED = 5,
        HIPBLAS_STATUS_INTERNAL_ERROR = 6,
        HIPBLAS_STATUS_NOT_SUPPORTED = 7,
        HIPBLAS_STATUS_ARCH_MISMATCH = 8,
        HIPBLAS_STATUS_HANDLE_IS_NULLPTR = 9,
        HIPBLAS_STATUS_INVALID_ENUM = 10,
        HIPBLAS_STATUS_UNKNOWN = 11,
    }

    #[repr(i32)]
    #[non_exhaustive]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub enum hipblasOperation_t {
        HIPBLAS_OP_N = 111,
        HIPBLAS_OP_T = 112,
        HIPBLAS_OP_C = 113,
    }

    #[repr(i32)]
    #[non_exhaustive]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub enum hipError {
        hipSuccess = 0,
        hipErrorInvalidValue = 1,
        hipErrorOutOfMemory = 2,
        hipErrorNotInitialized = 3,
        hipErrorDeinitialized = 4,
        hipErrorProfilerDisabled = 5,
        hipErrorProfilerNotInitialized = 6,
        hipErrorProfilerAlreadyStarted = 7,
        hipErrorProfilerAlreadyStopped = 8,
        hipErrorInvalidConfiguration = 9,
        hipErrorInvalidPitchValue = 12,
        hipErrorInvalidSymbol = 13,
        hipErrorInvalidDevicePointer = 17,
        hipErrorInvalidMemcpyDirection = 21,
        hipErrorInsufficientDriver = 35,
        hipErrorMissingConfiguration = 52,
        hipErrorPriorLaunchFailure = 53,
        hipErrorInvalidDeviceFunction = 98,
        hipErrorNoDevice = 100,
        hipErrorInvalidDevice = 101,
        hipErrorInvalidImage = 200,
        hipErrorInvalidContext = 201,
        hipErrorContextAlreadyCurrent = 202,
        hipErrorMapFailed = 205,
        hipErrorUnmapFailed = 206,
        hipErrorArrayIsMapped = 207,
        hipErrorAlreadyMapped = 208,
        hipErrorNoBinaryForGpu = 209,
        hipErrorAlreadyAcquired = 210,
        hipErrorNotMapped = 211,
        hipErrorNotMappedAsArray = 212,
        hipErrorNotMappedAsPointer = 213,
        hipErrorECCNotCorrectable = 214,
        hipErrorUnsupportedLimit = 215,
        hipErrorContextAlreadyInUse = 216,
        hipErrorPeerAccessUnsupported = 217,
        hipErrorInvalidKernelFile = 218,
        hipErrorInvalidGraphicsContext = 219,
        hipErrorInvalidSource = 300,
        hipErrorFileNotFound = 301,
        hipErrorSharedObjectSymbolNotFound = 302,
        hipErrorSharedObjectInitFailed = 303,
        hipErrorOperatingSystem = 304,
        hipErrorInvalidHandle = 400,
        hipErrorIllegalState = 401,
        hipErrorNotFound = 500,
        hipErrorNotReady = 600,
        hipErrorIllegalAddress = 700,
        hipErrorLaunchOutOfResources = 701,
        hipErrorLaunchTimeOut = 702,
        hipErrorPeerAccessAlreadyEnabled = 704,
        hipErrorPeerAccessNotEnabled = 705,
        hipErrorSetOnActiveProcess = 708,
        hipErrorContextIsDestroyed = 709,
        hipErrorAssert = 710,
        hipErrorHostMemoryAlreadyRegistered = 712,
        hipErrorHostMemoryNotRegistered = 713,
        hipErrorLaunchFailure = 719,
        hipErrorCooperativeLaunchTooLarge = 720,
        hipErrorNotSupported = 801,
        hipErrorStreamCaptureUnsupported = 900,
        hipErrorStreamCaptureInvalidated = 901,
        hipErrorStreamCaptureMerge = 902,
        hipErrorStreamCaptureUnmatched = 903,
        hipErrorStreamCaptureUnjoined = 904,
        hipErrorStreamCaptureIsolation = 905,
        hipErrorStreamCaptureImplicit = 906,
        hipErrorCapturedEvent = 907,
        hipErrorStreamCaptureWrongThread = 908,
        hipErrorGraphExecUpdateFailure = 910,
        hipErrorUnknown = 999,
        hipErrorRuntimeMemory = 1052,
        hipErrorRuntimeOther = 1053,
        hipErrorTbd = 1054,
    }
}
