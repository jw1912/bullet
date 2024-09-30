use std::os::raw::{c_char, c_int, c_longlong, c_uint, c_void};

pub const H2D: cudaMemcpyKind = cudaMemcpyKind::cudaMemcpyHostToDevice;
pub const D2H: cudaMemcpyKind = cudaMemcpyKind::cudaMemcpyDeviceToHost;
pub const D2D: cudaMemcpyKind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
pub const SUCCESS: cudaError_t = cudaError_t::cudaSuccess;
pub const CUBLAS_SUCCESS: cublasStatus_t = cublasStatus_t::CUBLAS_STATUS_SUCCESS;
pub const CUBLAS_OP_N: cublasOperation_t = cublasOperation_t::CUBLAS_OP_N;
pub const CUBLAS_OP_T: cublasOperation_t = cublasOperation_t::CUBLAS_OP_T;

pub type cudaError_t = cudaError;

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

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cudaMemoryType {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3,
}
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
pub struct CUuuid_st {
    pub bytes: [c_char; 16usize],
}
pub type CUuuid = CUuuid_st;
pub type cudaUUID_t = CUuuid_st;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaDeviceProp {
    pub name: [c_char; 256usize],
    pub uuid: cudaUUID_t,
    pub luid: [c_char; 8usize],
    pub luidDeviceNodeMask: c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: c_int,
    pub warpSize: c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: c_int,
    pub maxThreadsDim: [c_int; 3usize],
    pub maxGridSize: [c_int; 3usize],
    pub clockRate: c_int,
    pub totalConstMem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: c_int,
    pub multiProcessorCount: c_int,
    pub kernelExecTimeoutEnabled: c_int,
    pub integrated: c_int,
    pub canMapHostMemory: c_int,
    pub computeMode: c_int,
    pub maxTexture1D: c_int,
    pub maxTexture1DMipmap: c_int,
    pub maxTexture1DLinear: c_int,
    pub maxTexture2D: [c_int; 2usize],
    pub maxTexture2DMipmap: [c_int; 2usize],
    pub maxTexture2DLinear: [c_int; 3usize],
    pub maxTexture2DGather: [c_int; 2usize],
    pub maxTexture3D: [c_int; 3usize],
    pub maxTexture3DAlt: [c_int; 3usize],
    pub maxTextureCubemap: c_int,
    pub maxTexture1DLayered: [c_int; 2usize],
    pub maxTexture2DLayered: [c_int; 3usize],
    pub maxTextureCubemapLayered: [c_int; 2usize],
    pub maxSurface1D: c_int,
    pub maxSurface2D: [c_int; 2usize],
    pub maxSurface3D: [c_int; 3usize],
    pub maxSurface1DLayered: [c_int; 2usize],
    pub maxSurface2DLayered: [c_int; 3usize],
    pub maxSurfaceCubemap: c_int,
    pub maxSurfaceCubemapLayered: [c_int; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: c_int,
    pub ECCEnabled: c_int,
    pub pciBusID: c_int,
    pub pciDeviceID: c_int,
    pub pciDomainID: c_int,
    pub tccDriver: c_int,
    pub asyncEngineCount: c_int,
    pub unifiedAddressing: c_int,
    pub memoryClockRate: c_int,
    pub memoryBusWidth: c_int,
    pub l2CacheSize: c_int,
    pub persistingL2CacheMaxSize: c_int,
    pub maxThreadsPerMultiProcessor: c_int,
    pub streamPrioritiesSupported: c_int,
    pub globalL1CacheSupported: c_int,
    pub localL1CacheSupported: c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: c_int,
    pub managedMemory: c_int,
    pub isMultiGpuBoard: c_int,
    pub multiGpuBoardGroupID: c_int,
    pub hostNativeAtomicSupported: c_int,
    pub singleToDoublePrecisionPerfRatio: c_int,
    pub pageableMemoryAccess: c_int,
    pub concurrentManagedAccess: c_int,
    pub computePreemptionSupported: c_int,
    pub canUseHostPointerForRegisteredMem: c_int,
    pub cooperativeLaunch: c_int,
    pub cooperativeMultiDeviceLaunch: c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: c_int,
    pub directManagedMemAccessFromHost: c_int,
    pub maxBlocksPerMultiProcessor: c_int,
    pub accessPolicyMaxWindowSize: c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: c_int,
    pub sparseCudaArraySupported: c_int,
    pub hostRegisterReadOnlySupported: c_int,
    pub timelineSemaphoreInteropSupported: c_int,
    pub memoryPoolsSupported: c_int,
    pub gpuDirectRDMASupported: c_int,
    pub gpuDirectRDMAFlushWritesOptions: c_uint,
    pub gpuDirectRDMAWritesOrdering: c_int,
    pub memoryPoolSupportedHandleTypes: c_uint,
    pub deferredMappingCudaArraySupported: c_int,
    pub ipcEventSupported: c_int,
    pub clusterLaunch: c_int,
    pub unifiedFunctionPointers: c_int,
    pub reserved2: [c_int; 2usize],
    pub reserved1: [c_int; 1usize],
    pub reserved: [c_int; 60usize],
}

extern "C" {
    pub fn cudaDeviceReset() -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaGetLastError() -> cudaError_t;
    pub fn cudaPeekAtLastError() -> cudaError_t;
    pub fn cudaGetErrorName(error: cudaError_t) -> *const c_char;
    pub fn cudaGetErrorString(error: cudaError_t) -> *const c_char;
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    pub fn cudaGetDeviceProperties_v2(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;
    pub fn cudaChooseDevice(device: *mut c_int, prop: *const cudaDeviceProp) -> cudaError_t;
    pub fn cudaInitDevice(device: c_int, deviceFlags: c_uint, flags: c_uint) -> cudaError_t;
    pub fn cudaSetDevice(device: c_int) -> cudaError_t;
    pub fn cudaGetDevice(device: *mut c_int) -> cudaError_t;
    pub fn cudaSetValidDevices(device_arr: *mut c_int, len: c_int) -> cudaError_t;
    pub fn cudaSetDeviceFlags(flags: c_uint) -> cudaError_t;
    pub fn cudaGetDeviceFlags(flags: *mut c_uint) -> cudaError_t;
    pub fn cudaMallocManaged(devPtr: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t;
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaMallocPitch(devPtr: *mut *mut c_void, pitch: *mut usize, width: usize, height: usize) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    pub fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t;
    pub fn cudaMemset(devPtr: *mut c_void, value: c_int, count: usize) -> cudaError_t;
}

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cublasStatus_t {
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
pub enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
    CUBLAS_OP_CONJG = 3,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cublasContext {
    _unused: [u8; 0],
}
pub type cublasHandle_t = *mut cublasContext;

extern "C" {
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;

    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;

    pub fn cublasSnrm2_v2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f32,
        incx: c_int,
        result: *mut f32,
    ) -> cublasStatus_t;

    pub fn cublasSnrm2_v2_64(
        handle: cublasHandle_t,
        n: i64,
        x: *const f32,
        incx: i64,
        result: *mut f32,
    ) -> cublasStatus_t;

    pub fn cublasSdot_v2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f32,
        incx: c_int,
        y: *const f32,
        incy: c_int,
        result: *mut f32,
    ) -> cublasStatus_t;

    pub fn cublasSdot_v2_64(
        handle: cublasHandle_t,
        n: i64,
        x: *const f32,
        incx: i64,
        y: *const f32,
        incy: i64,
        result: *mut f32,
    ) -> cublasStatus_t;

    pub fn cublasSscal_v2(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const f32,
        x: *mut f32,
        incx: c_int,
    ) -> cublasStatus_t;

    pub fn cublasSscal_v2_64(
        handle: cublasHandle_t,
        n: i64,
        alpha: *const f32,
        x: *mut f32,
        incx: i64,
    ) -> cublasStatus_t;

    pub fn cublasSaxpy_v2(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const f32,
        x: *const f32,
        incx: c_int,
        y: *mut f32,
        incy: c_int,
    ) -> cublasStatus_t;

    pub fn cublasSaxpy_v2_64(
        handle: cublasHandle_t,
        n: i64,
        alpha: *const f32,
        x: *const f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
    ) -> cublasStatus_t;

    pub fn cublasScopy_v2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f32,
        incx: c_int,
        y: *mut f32,
        incy: c_int,
    ) -> cublasStatus_t;

    pub fn cublasScopy_v2_64(
        handle: cublasHandle_t,
        n: i64,
        x: *const f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
    ) -> cublasStatus_t;

    pub fn cublasSswap_v2(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut f32,
        incx: c_int,
        y: *mut f32,
        incy: c_int,
    ) -> cublasStatus_t;

    pub fn cublasSswap_v2_64(
        handle: cublasHandle_t,
        n: i64,
        x: *mut f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
    ) -> cublasStatus_t;

    pub fn cublasSasum_v2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f32,
        incx: c_int,
        result: *mut f32,
    ) -> cublasStatus_t;

    pub fn cublasSasum_v2_64(
        handle: cublasHandle_t,
        n: i64,
        x: *const f32,
        incx: i64,
        result: *mut f32,
    ) -> cublasStatus_t;

    pub fn cublasSgemv_v2(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: c_int,
        n: c_int,
        alpha: *const f32,
        A: *const f32,
        lda: c_int,
        x: *const f32,
        incx: c_int,
        beta: *const f32,
        y: *mut f32,
        incy: c_int,
    ) -> cublasStatus_t;

    pub fn cublasSgemv_v2_64(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: i64,
        n: i64,
        alpha: *const f32,
        A: *const f32,
        lda: i64,
        x: *const f32,
        incx: i64,
        beta: *const f32,
        y: *mut f32,
        incy: i64,
    ) -> cublasStatus_t;

    pub fn cublasSgemvStridedBatched(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: c_int,
        n: c_int,
        alpha: *const f32,
        A: *const f32,
        lda: c_int,
        strideA: c_longlong,
        x: *const f32,
        incx: c_int,
        stridex: c_longlong,
        beta: *const f32,
        y: *mut f32,
        incy: c_int,
        stridey: c_longlong,
        batchCount: c_int,
    ) -> cublasStatus_t;

    pub fn cublasSgemvStridedBatched_64(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: i64,
        n: i64,
        alpha: *const f32,
        A: *const f32,
        lda: i64,
        strideA: c_longlong,
        x: *const f32,
        incx: i64,
        stridex: c_longlong,
        beta: *const f32,
        y: *mut f32,
        incy: i64,
        stridey: c_longlong,
        batchCount: i64,
    ) -> cublasStatus_t;

    pub fn cublasSgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
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
    ) -> cublasStatus_t;

    pub fn cublasSgemm_v2_64(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i64,
        n: i64,
        k: i64,
        alpha: *const f32,
        A: *const f32,
        lda: i64,
        B: *const f32,
        ldb: i64,
        beta: *const f32,
        C: *mut f32,
        ldc: i64,
    ) -> cublasStatus_t;

    pub fn cublasSgemmBatched(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        Aarray: *const *const f32,
        lda: c_int,
        Barray: *const *const f32,
        ldb: c_int,
        beta: *const f32,
        Carray: *const *mut f32,
        ldc: c_int,
        batchCount: c_int,
    ) -> cublasStatus_t;

    pub fn cublasSgemmBatched_64(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i64,
        n: i64,
        k: i64,
        alpha: *const f32,
        Aarray: *const *const f32,
        lda: i64,
        Barray: *const *const f32,
        ldb: i64,
        beta: *const f32,
        Carray: *const *mut f32,
        ldc: i64,
        batchCount: i64,
    ) -> cublasStatus_t;

    pub fn cublasSgemmStridedBatched(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
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
    ) -> cublasStatus_t;

    pub fn cublasSgemmStridedBatched_64(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i64,
        n: i64,
        k: i64,
        alpha: *const f32,
        A: *const f32,
        lda: i64,
        strideA: c_longlong,
        B: *const f32,
        ldb: i64,
        strideB: c_longlong,
        beta: *const f32,
        C: *mut f32,
        ldc: i64,
        strideC: c_longlong,
        batchCount: i64,
    ) -> cublasStatus_t;

    pub fn cublasSgeam(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        alpha: *const f32,
        A: *const f32,
        lda: c_int,
        beta: *const f32,
        B: *const f32,
        ldb: c_int,
        C: *mut f32,
        ldc: c_int,
    ) -> cublasStatus_t;

    pub fn cublasSgeam_64(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i64,
        n: i64,
        alpha: *const f32,
        A: *const f32,
        lda: i64,
        beta: *const f32,
        B: *const f32,
        ldb: i64,
        C: *mut f32,
        ldc: i64,
    ) -> cublasStatus_t;
}
