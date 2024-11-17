#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cudnnStatus_t {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1,
    CUDNN_STATUS_ALLOC_FAILED = 2,
    CUDNN_STATUS_BAD_PARAM = 3,
    CUDNN_STATUS_INTERNAL_ERROR = 4,
    CUDNN_STATUS_INVALID_VALUE = 5,
    CUDNN_STATUS_ARCH_MISMATCH = 6,
    CUDNN_STATUS_MAPPING_ERROR = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
    CUDNN_STATUS_LICENSE_ERROR = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13,
    CUDNN_STATUS_VERSION_MISMATCH = 14,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudnnContext;
pub type cudnnHandle_t = *mut cudnnContext;

#[cfg(feature = "cudnn")]
pub use cudnn::*;

#[cfg(feature = "cudnn")]
mod cudnn {
    use super::*;

    #[rustfmt::skip]
    extern "C" {
        pub fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;
        pub fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;
    }
}

#[cfg(not(feature = "cudnn"))]
pub use fallback::*;

#[cfg(not(feature = "cudnn"))]
#[allow(clippy::undocumented_unsafe_blocks)]
mod fallback {
    use super::*;

    pub unsafe fn cudnnCreate(_: *mut cudnnHandle_t) -> cudnnStatus_t {
        cudnnStatus_t::CUDNN_STATUS_SUCCESS
    }

    pub unsafe fn cudnnDestroy(_: cudnnHandle_t) -> cudnnStatus_t {
        cudnnStatus_t::CUDNN_STATUS_SUCCESS
    }
}
