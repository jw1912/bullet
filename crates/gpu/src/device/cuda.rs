//! Minimal wrapper around the CUDA runtime

use std::ffi::{c_int, c_void};

use super::bindings::{Dim3, GpuBindings, MemcpyKind};

/// Marker for the CUDA runtime
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Cuda;

/// Error type for the CUDA runtime
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaError {
    Runtime(raw::cudaError),
    Blas(raw::cublasStatus),
    Message(String),
}

impl From<String> for CudaError {
    fn from(value: String) -> Self {
        Self::Message(value)
    }
}

mod sealed {
    use super::raw;

    #[repr(transparent)]
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct CudaStream(pub(super) raw::cudaStream);

    #[repr(transparent)]
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct CudaBlasHandle(pub(super) raw::cublasHandle);
}

use sealed::{CudaBlasHandle, CudaStream};

type CudaResult = Result<(), CudaError>;

impl GpuBindings for Cuda {
    type E = CudaError;
    type S = CudaStream;
    type B = CudaBlasHandle;

    unsafe fn device_init(device: c_int) -> CudaResult {
        unsafe { err(raw::cudaInitDevice(device, 0, 0)) }
    }

    unsafe fn device_set(device: c_int) -> CudaResult {
        unsafe { err(raw::cudaSetDevice(device)) }
    }

    unsafe fn stream_create(stream: *mut CudaStream) -> CudaResult {
        unsafe { err(raw::cudaStreamCreate(stream.cast())) }
    }

    unsafe fn stream_destroy(stream: CudaStream) -> CudaResult {
        unsafe { err(raw::cudaStreamDestroy(stream.0)) }
    }

    unsafe fn stream_sync(stream: CudaStream) -> CudaResult {
        unsafe { err(raw::cudaStreamSynchronize(stream.0)) }
    }

    unsafe fn stream_malloc(stream: CudaStream, dev_ptr: *mut *mut c_void, bytes: usize) -> CudaResult {
        unsafe { err(raw::cudaMallocAsync(dev_ptr, bytes, stream.0)) }
    }

    unsafe fn stream_free(stream: CudaStream, dev_ptr: *mut c_void) -> CudaResult {
        unsafe { err(raw::cudaFreeAsync(dev_ptr, stream.0)) }
    }

    unsafe fn stream_memset(stream: Self::S, dev_ptr: *mut c_void, bytes: usize, value: u8) -> Result<(), Self::E> {
        unsafe { err(raw::cudaMemsetAsync(dev_ptr, value as c_int, bytes, stream.0)) }
    }

    unsafe fn stream_memcpy(
        stream: CudaStream,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: MemcpyKind,
    ) -> CudaResult {
        let kind = match kind {
            MemcpyKind::HostToHost => raw::cudaMemcpyKind::cudaMemcpyHostToHost,
            MemcpyKind::DeviceToDevice => raw::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            MemcpyKind::DeviceToHost => raw::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            MemcpyKind::HostToDevice => raw::cudaMemcpyKind::cudaMemcpyHostToDevice,
            MemcpyKind::Default => raw::cudaMemcpyKind::cudaMemcpyDefault,
        };

        unsafe { err(raw::cudaMemcpyAsync(dst, src, bytes, kind, stream.0)) }
    }

    unsafe fn stream_launch_kernel(
        stream: CudaStream,
        func: *const c_void,
        gdim: Dim3,
        bdim: Dim3,
        args: *mut *mut c_void,
        smem: usize,
    ) -> CudaResult {
        let gdim = raw::dim3 { x: gdim.x, y: gdim.y, z: gdim.z };
        let bdim = raw::dim3 { x: bdim.x, y: bdim.y, z: bdim.z };
        unsafe { err(raw::cudaLaunchKernel(func, gdim, bdim, args, smem, stream.0)) }
    }
}

fn err(value: raw::cudaError) -> CudaResult {
    if value != raw::cudaError::cudaSuccess { Err(CudaError::Runtime(value)) } else { Ok(()) }
}

fn status(value: raw::cublasStatus) -> CudaResult {
    if value != raw::cublasStatus::CUBLAS_STATUS_SUCCESS { Err(CudaError::Blas(value)) } else { Ok(()) }
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
        pub fn cudaInitDevice(device: c_int, deviceFlags: c_uint, flags: c_uint) -> cudaError;
        pub fn cudaSetDevice(device: c_int) -> cudaError;
        pub fn cudaStreamCreate(stream: *mut cudaStream) -> cudaError;
        pub fn cudaStreamDestroy(stream: cudaStream) -> cudaError;
        pub fn cudaStreamSynchronize(stream: cudaStream) -> cudaError;
        pub fn cudaMallocAsync(devPtr: *mut *mut c_void, size: usize, stream: cudaStream) -> cudaError;
        pub fn cudaFreeAsync(devPtr: *mut c_void, stream: cudaStream) -> cudaError;
        pub fn cudaMemsetAsync (devPtr: *mut c_void, value: c_int, count: usize, stream: cudaStream) -> cudaError;
        pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: cudaMemcpyKind, stream: cudaStream) -> cudaError;
        pub fn cudaLaunchKernel(func: *const c_void, gridDim: dim3, blockDim: dim3, args: *mut *mut c_void, sharedMem: usize, stream: cudaStream) -> cudaError;
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct CudaStreamOpaque([u8; 0]);
    pub type cudaStream = *mut CudaStreamOpaque;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct CublasContextOpaque([u8; 0]);
    pub type cublasHandle = *mut CublasContextOpaque;

    #[repr(i32)]
    #[non_exhaustive]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub enum cudaMemcpyKind {
        cudaMemcpyHostToHost = 0,
        cudaMemcpyHostToDevice = 1,
        cudaMemcpyDeviceToHost = 2,
        cudaMemcpyDeviceToDevice = 3,
        cudaMemcpyDefault = 4,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct dim3 {
        pub x: ::std::os::raw::c_uint,
        pub y: ::std::os::raw::c_uint,
        pub z: ::std::os::raw::c_uint,
    }

    #[repr(i32)]
    #[non_exhaustive]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub enum cublasStatus {
        CUBLAS_STATUS_SUCCESS = 0,
        CUBLAS_STATUS_NOT_INITIALIZED = 1,
        CUBLAS_STATUS_ALLOC_FAILED = 3,
        CUBLAS_STATUS_INVALID_VALUE = 7,
        CUBLAS_STATUS_ARCH_MISMATCH = 8,
        CUBLAS_STATUS_MAPPING_ERROR = 11,
        CUBLAS_STATUS_EXECUTION_FAILED = 13,
        CUBLAS_STATUS_INTERNAL_ERROR = 14,
        CUBLAS_STATUS_NOT_SUPPORTED = 15,
        CUBLAS_STATUS_LICENSE_ERROR = 16,
    }

    #[repr(i32)]
    #[non_exhaustive]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub enum cublasOperation {
        CUBLAS_OP_N = 0,
        CUBLAS_OP_T = 1,
        CUBLAS_OP_C = 2,
        CUBLAS_OP_CONJG = 3,
    }

    #[repr(i32)]
    #[non_exhaustive]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub enum cudaError {
        cudaSuccess = 0,
        cudaErrorInvalidValue = 1,
        cudaErrorMemoryAllocation = 2,
        cudaErrorInitializationError = 3,
        cudaErrorCudartUnloading = 4,
        cudaErrorProfilerDisabled = 5,
        cudaErrorProfilerNotInitialized = 6,
        cudaErrorProfilerAlreadyStarted = 7,
        cudaErrorProfilerAlreadyStopped = 8,
        cudaErrorInvalidConfiguration = 9,
        cudaErrorInvalidPitchValue = 12,
        cudaErrorInvalidSymbol = 13,
        cudaErrorInvalidHostPointer = 16,
        cudaErrorInvalidDevicePointer = 17,
        cudaErrorInvalidTexture = 18,
        cudaErrorInvalidTextureBinding = 19,
        cudaErrorInvalidChannelDescriptor = 20,
        cudaErrorInvalidMemcpyDirection = 21,
        cudaErrorAddressOfConstant = 22,
        cudaErrorTextureFetchFailed = 23,
        cudaErrorTextureNotBound = 24,
        cudaErrorSynchronizationError = 25,
        cudaErrorInvalidFilterSetting = 26,
        cudaErrorInvalidNormSetting = 27,
        cudaErrorMixedDeviceExecution = 28,
        cudaErrorNotYetImplemented = 31,
        cudaErrorMemoryValueTooLarge = 32,
        cudaErrorStubLibrary = 34,
        cudaErrorInsufficientDriver = 35,
        cudaErrorCallRequiresNewerDriver = 36,
        cudaErrorInvalidSurface = 37,
        cudaErrorDuplicateVariableName = 43,
        cudaErrorDuplicateTextureName = 44,
        cudaErrorDuplicateSurfaceName = 45,
        cudaErrorDevicesUnavailable = 46,
        cudaErrorIncompatibleDriverContext = 49,
        cudaErrorMissingConfiguration = 52,
        cudaErrorPriorLaunchFailure = 53,
        cudaErrorLaunchMaxDepthExceeded = 65,
        cudaErrorLaunchFileScopedTex = 66,
        cudaErrorLaunchFileScopedSurf = 67,
        cudaErrorSyncDepthExceeded = 68,
        cudaErrorLaunchPendingCountExceeded = 69,
        cudaErrorInvalidDeviceFunction = 98,
        cudaErrorNoDevice = 100,
        cudaErrorInvalidDevice = 101,
        cudaErrorDeviceNotLicensed = 102,
        cudaErrorSoftwareValidityNotEstablished = 103,
        cudaErrorStartupFailure = 127,
        cudaErrorInvalidKernelImage = 200,
        cudaErrorDeviceUninitialized = 201,
        cudaErrorMapBufferObjectFailed = 205,
        cudaErrorUnmapBufferObjectFailed = 206,
        cudaErrorArrayIsMapped = 207,
        cudaErrorAlreadyMapped = 208,
        cudaErrorNoKernelImageForDevice = 209,
        cudaErrorAlreadyAcquired = 210,
        cudaErrorNotMapped = 211,
        cudaErrorNotMappedAsArray = 212,
        cudaErrorNotMappedAsPointer = 213,
        cudaErrorECCUncorrectable = 214,
        cudaErrorUnsupportedLimit = 215,
        cudaErrorDeviceAlreadyInUse = 216,
        cudaErrorPeerAccessUnsupported = 217,
        cudaErrorInvalidPtx = 218,
        cudaErrorInvalidGraphicsContext = 219,
        cudaErrorNvlinkUncorrectable = 220,
        cudaErrorJitCompilerNotFound = 221,
        cudaErrorUnsupportedPtxVersion = 222,
        cudaErrorJitCompilationDisabled = 223,
        cudaErrorUnsupportedExecAffinity = 224,
        cudaErrorUnsupportedDevSideSync = 225,
        cudaErrorInvalidSource = 300,
        cudaErrorFileNotFound = 301,
        cudaErrorSharedObjectSymbolNotFound = 302,
        cudaErrorSharedObjectInitFailed = 303,
        cudaErrorOperatingSystem = 304,
        cudaErrorInvalidResourceHandle = 400,
        cudaErrorIllegalState = 401,
        cudaErrorLossyQuery = 402,
        cudaErrorSymbolNotFound = 500,
        cudaErrorNotReady = 600,
        cudaErrorIllegalAddress = 700,
        cudaErrorLaunchOutOfResources = 701,
        cudaErrorLaunchTimeout = 702,
        cudaErrorLaunchIncompatibleTexturing = 703,
        cudaErrorPeerAccessAlreadyEnabled = 704,
        cudaErrorPeerAccessNotEnabled = 705,
        cudaErrorSetOnActiveProcess = 708,
        cudaErrorContextIsDestroyed = 709,
        cudaErrorAssert = 710,
        cudaErrorTooManyPeers = 711,
        cudaErrorHostMemoryAlreadyRegistered = 712,
        cudaErrorHostMemoryNotRegistered = 713,
        cudaErrorHardwareStackError = 714,
        cudaErrorIllegalInstruction = 715,
        cudaErrorMisalignedAddress = 716,
        cudaErrorInvalidAddressSpace = 717,
        cudaErrorInvalidPc = 718,
        cudaErrorLaunchFailure = 719,
        cudaErrorCooperativeLaunchTooLarge = 720,
        cudaErrorNotPermitted = 800,
        cudaErrorNotSupported = 801,
        cudaErrorSystemNotReady = 802,
        cudaErrorSystemDriverMismatch = 803,
        cudaErrorCompatNotSupportedOnDevice = 804,
        cudaErrorMpsConnectionFailed = 805,
        cudaErrorMpsRpcFailure = 806,
        cudaErrorMpsServerNotReady = 807,
        cudaErrorMpsMaxClientsReached = 808,
        cudaErrorMpsMaxConnectionsReached = 809,
        cudaErrorMpsClientTerminated = 810,
        cudaErrorCdpNotSupported = 811,
        cudaErrorCdpVersionMismatch = 812,
        cudaErrorStreamCaptureUnsupported = 900,
        cudaErrorStreamCaptureInvalidated = 901,
        cudaErrorStreamCaptureMerge = 902,
        cudaErrorStreamCaptureUnmatched = 903,
        cudaErrorStreamCaptureUnjoined = 904,
        cudaErrorStreamCaptureIsolation = 905,
        cudaErrorStreamCaptureImplicit = 906,
        cudaErrorCapturedEvent = 907,
        cudaErrorStreamCaptureWrongThread = 908,
        cudaErrorTimeout = 909,
        cudaErrorGraphExecUpdateFailure = 910,
        cudaErrorExternalDevice = 911,
        cudaErrorInvalidClusterSize = 912,
        cudaErrorFunctionNotLoaded = 913,
        cudaErrorInvalidResourceType = 914,
        cudaErrorInvalidResourceConfiguration = 915,
        cudaErrorUnknown = 999,
        cudaErrorApiFailureBase = 10000,
    }
}
