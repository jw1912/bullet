use std::os::raw::{c_char, c_int, c_longlong, c_uint, c_void};

pub const H2D: hipMemcpyKind = hipMemcpyKind::hipMemcpyHostToDevice;
pub const D2H: hipMemcpyKind = hipMemcpyKind::hipMemcpyDeviceToHost;
pub const D2D: hipMemcpyKind = hipMemcpyKind::hipMemcpyDeviceToDevice;
pub const SUCCESS: hipError_t = hipError_t::hipSuccess;
pub const CUBLAS_SUCCESS: hipblasStatus_t = hipblasStatus_t::HIPBLAS_STATUS_SUCCESS;
pub const CUBLAS_OP_N: hipblasOperation_t = hipblasOperation_t::HIPBLAS_OP_N;
pub const CUBLAS_OP_T: hipblasOperation_t = hipblasOperation_t::HIPBLAS_OP_T;

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

#[rustfmt::skip]
extern "C" {
    pub fn hipDeviceReset() -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;
    pub fn hipGetLastError() -> hipError_t;
    pub fn hipPeekAtLastError() -> hipError_t;
    pub fn hipGetErrorName(error: hipError_t) -> *const c_char;
    pub fn hipGetErrorString(error: hipError_t) -> *const c_char;
    pub fn hipMalloc(devPtr: *mut *mut c_void, size: usize) -> hipError_t;
    pub fn hipFree(devPtr: *mut c_void) -> hipError_t;
    pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: hipMemcpyKind) -> hipError_t;
    pub fn hipMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: hipMemcpyKind, stream: hipStream_t) -> hipError_t;
    pub fn hipMemset(devPtr: *mut c_void, value: c_int, count: usize) -> hipError_t;
    pub fn hipStreamCreateWithFlags(stream: *mut hipStream_t, flags: ::std::os::raw::c_uint) -> hipError_t;
    pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ihipStream_t {
    _unused: [u8; 0],
}
pub type hipStream_t = *mut ihipStream_t;

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasStatus_t {
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

pub type hipblasHandle_t = *mut c_void;

#[rustfmt::skip]
extern "C" {
    pub fn hipblasCreate(handle: *mut hipblasHandle_t) -> hipblasStatus_t;
    pub fn hipblasDestroy(handle: hipblasHandle_t) -> hipblasStatus_t;
    pub fn hipblasSaxpy(handle: hipblasHandle_t, n: c_int, alpha: *const f32, x: *const f32, incx: c_int, y: *mut f32, incy: c_int) -> hipblasStatus_t;
    pub fn hipblasSgemm(handle: hipblasHandle_t, transa: hipblasOperation_t, transb: hipblasOperation_t, m: c_int, n: c_int, k: c_int, alpha: *const f32, A: *const f32, lda: c_int, B: *const f32, ldb: c_int, beta: *const f32, C: *mut f32, ldc: c_int) -> hipblasStatus_t;
    pub fn hipblasSgeam(handle: hipblasHandle_t, transa: hipblasOperation_t, transb: hipblasOperation_t, m: c_int, n: c_int, alpha: *const f32, A: *const f32, lda: c_int, beta: *const f32, B: *const f32, ldb: c_int, C: *mut f32, ldc: c_int) -> hipblasStatus_t;
    pub fn hipblasSger(handle: hipblasHandle_t, m: c_int, n: c_int, alpha: *const f32, x: *const f32, incx: c_int, y: *const f32, incy: c_int, A: *mut f32, lda: c_int) -> hipblasStatus_t;
    pub fn hipblasSgemmStridedBatched(handle: hipblasHandle_t, transA: hipblasOperation_t, transB: hipblasOperation_t, m: c_int, n: c_int, k: c_int, alpha: *const f32, AP: *const f32, lda: c_int, strideA: c_longlong, BP: *const f32, ldb: c_int, strideB: c_longlong, beta: *const f32, CP: *mut f32, ldc: c_int, strideC: c_longlong, batchCount: c_int) -> hipblasStatus_t;
}

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipError_t {
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
