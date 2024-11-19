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
pub struct cudnnContext {
    _unused: [u8; 0],
}

pub type cudnnHandle_t = *mut cudnnContext;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudnnConvolutionStruct {
    _unused: [u8; 0],
}

pub type cudnnConvolutionDescriptor_t = *mut cudnnConvolutionStruct;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudnnTensorStruct {
    _unused: [u8; 0],
}

pub type cudnnTensorDescriptor_t = *mut cudnnTensorStruct;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudnnFilterStruct {
    _unused: [u8; 0],
}

pub type cudnnFilterDescriptor_t = *mut cudnnFilterStruct;

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cudnnConvolutionFwdAlgo_t {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8,
}

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cudnnConvolutionBwdFilterAlgo_t {
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7,
}

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cudnnConvolutionBwdDataAlgo_t {
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6,
}

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cudnnTensorFormat_t {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
    CUDNN_TENSOR_NCHW_VECT_C = 2,
}

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cudnnDataType_t {
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF = 2,
    CUDNN_DATA_INT8 = 3,
    CUDNN_DATA_INT32 = 4,
    CUDNN_DATA_INT8x4 = 5,
    CUDNN_DATA_UINT8 = 6,
    CUDNN_DATA_UINT8x4 = 7,
    CUDNN_DATA_INT8x32 = 8,
    CUDNN_DATA_BFLOAT16 = 9,
    CUDNN_DATA_INT64 = 10,
    CUDNN_DATA_BOOLEAN = 11,
    CUDNN_DATA_FP8_E4M3 = 12,
    CUDNN_DATA_FP8_E5M2 = 13,
    CUDNN_DATA_FAST_FLOAT_FOR_FP8 = 14,
}

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cudnnConvolutionMode_t {
    CUDNN_CONVOLUTION = 0,
    CUDNN_CROSS_CORRELATION = 1,
}

#[cfg(feature = "cudnn")]
pub use cudnn::*;

#[cfg(feature = "cudnn")]
mod cudnn {
    use std::os::raw::{c_int, c_void};

    use super::*;

    #[rustfmt::skip]
    extern "C" {
        pub fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;
        pub fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;
        pub fn cudnnCreateTensorDescriptor(tensorDesc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t;
        pub fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) -> cudnnStatus_t;
        pub fn cudnnSetTensor4dDescriptor(
            tensorDesc: cudnnTensorDescriptor_t,
            format: cudnnTensorFormat_t,
            dataType: cudnnDataType_t,
            n: c_int,
            c: c_int,
            h: c_int,
            w: c_int,
        ) -> cudnnStatus_t;
        pub fn cudnnCreateConvolutionDescriptor(convDesc: *mut cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
        pub fn cudnnDestroyConvolutionDescriptor(convDesc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
        pub fn cudnnSetConvolution2dDescriptor(
            convDesc: cudnnConvolutionDescriptor_t,
            pad_h: c_int,
            pad_w: c_int,
            u: c_int,
            v: c_int,
            dilation_h: c_int,
            dilation_w: c_int,
            mode: cudnnConvolutionMode_t,
            computeType: cudnnDataType_t,
        ) -> cudnnStatus_t;
        pub fn cudnnCreateFilterDescriptor(filterDesc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t;
        pub fn cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor_t) -> cudnnStatus_t;
        pub fn cudnnSetFilter4dDescriptor(
            filterDesc: cudnnFilterDescriptor_t,
            dataType: cudnnDataType_t,
            format: cudnnTensorFormat_t,
            k: c_int,
            c: c_int,
            h: c_int,
            w: c_int,
        ) -> cudnnStatus_t;
        pub fn cudnnConvolutionForward(
            handle: cudnnHandle_t,
            alpha: *const c_void,
            xDesc: cudnnTensorDescriptor_t,
            x: *const c_void,
            wDesc: cudnnFilterDescriptor_t,
            w: *const c_void,
            convDesc: cudnnConvolutionDescriptor_t,
            algo: cudnnConvolutionFwdAlgo_t,
            workSpace: *mut c_void,
            workSpaceSizeInBytes: usize,
            beta: *const c_void,
            yDesc: cudnnTensorDescriptor_t,
            y: *mut c_void,
        ) -> cudnnStatus_t;
        pub fn cudnnConvolutionBackwardData(
            handle: cudnnHandle_t,
            alpha: *const c_void,
            wDesc: cudnnFilterDescriptor_t,
            w: *const c_void,
            dyDesc: cudnnTensorDescriptor_t,
            dy: *const c_void,
            convDesc: cudnnConvolutionDescriptor_t,
            algo: cudnnConvolutionBwdDataAlgo_t,
            workSpace: *mut c_void,
            workSpaceSizeInBytes: usize,
            beta: *const c_void,
            dxDesc: cudnnTensorDescriptor_t,
            dx: *mut c_void,
        ) -> cudnnStatus_t;
        pub fn cudnnConvolutionBackwardFilter(
            handle: cudnnHandle_t,
            alpha: *const c_void,
            xDesc: cudnnTensorDescriptor_t,
            x: *const c_void,
            dyDesc: cudnnTensorDescriptor_t,
            dy: *const c_void,
            convDesc: cudnnConvolutionDescriptor_t,
            algo: cudnnConvolutionBwdFilterAlgo_t,
            workSpace: *mut c_void,
            workSpaceSizeInBytes: usize,
            beta: *const c_void,
            dwDesc: cudnnFilterDescriptor_t,
            dw: *mut c_void,
        ) -> cudnnStatus_t;
    }
}

#[cfg(not(feature = "cudnn"))]
pub use fallback::*;

#[cfg(not(feature = "cudnn"))]
#[allow(clippy::undocumented_unsafe_blocks)]
#[allow(clippy::too_many_arguments)]
mod fallback {
    use std::os::raw::{c_int, c_void};

    use super::*;

    pub unsafe fn cudnnCreate(_: *mut cudnnHandle_t) -> cudnnStatus_t {
        cudnnStatus_t::CUDNN_STATUS_SUCCESS
    }

    pub unsafe fn cudnnDestroy(_: cudnnHandle_t) -> cudnnStatus_t {
        cudnnStatus_t::CUDNN_STATUS_SUCCESS
    }

    pub unsafe fn cudnnCreateTensorDescriptor(tensorDesc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t {
        unimplemented!("Convolution is not implemented without cuDNN!");
    }

    pub unsafe fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) -> cudnnStatus_t {
        unimplemented!("Convolution is not implemented without cuDNN!");
    }

    pub unsafe fn cudnnSetTensor4dDescriptor(
        tensorDesc: cudnnTensorDescriptor_t,
        format: cudnnTensorFormat_t,
        dataType: cudnnDataType_t,
        n: c_int,
        c: c_int,
        h: c_int,
        w: c_int,
    ) -> cudnnStatus_t {
        unimplemented!("Convolution is not implemented without cuDNN!");
    }

    pub unsafe fn cudnnCreateConvolutionDescriptor(convDesc: *mut cudnnConvolutionDescriptor_t) -> cudnnStatus_t {
        unimplemented!("Convolution is not implemented without cuDNN!");
    }

    pub unsafe fn cudnnDestroyConvolutionDescriptor(convDesc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t {
        unimplemented!("Convolution is not implemented without cuDNN!");
    }

    pub unsafe fn cudnnSetConvolution2dDescriptor(
        convDesc: cudnnConvolutionDescriptor_t,
        pad_h: c_int,
        pad_w: c_int,
        u: c_int,
        v: c_int,
        dilation_h: c_int,
        dilation_w: c_int,
        mode: cudnnConvolutionMode_t,
        computeType: cudnnDataType_t,
    ) -> cudnnStatus_t {
        unimplemented!("Convolution is not implemented without cuDNN!");
    }

    pub unsafe fn cudnnCreateFilterDescriptor(filterDesc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t {
        unimplemented!("Convolution is not implemented without cuDNN!");
    }

    pub unsafe fn cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor_t) -> cudnnStatus_t {
        unimplemented!("Convolution is not implemented without cuDNN!");
    }

    pub unsafe fn cudnnSetFilter4dDescriptor(
        filterDesc: cudnnFilterDescriptor_t,
        dataType: cudnnDataType_t,
        format: cudnnTensorFormat_t,
        k: c_int,
        c: c_int,
        h: c_int,
        w: c_int,
    ) -> cudnnStatus_t {
        unimplemented!("Convolution is not implemented without cuDNN!");
    }

    pub unsafe fn cudnnConvolutionForward(
        handle: cudnnHandle_t,
        alpha: *const c_void,
        xDesc: cudnnTensorDescriptor_t,
        x: *const c_void,
        wDesc: cudnnFilterDescriptor_t,
        w: *const c_void,
        convDesc: cudnnConvolutionDescriptor_t,
        algo: cudnnConvolutionFwdAlgo_t,
        workSpace: *mut c_void,
        workSpaceSizeInBytes: usize,
        beta: *const c_void,
        yDesc: cudnnTensorDescriptor_t,
        y: *mut c_void,
    ) -> cudnnStatus_t {
        unimplemented!("Convolution is not implemented without cuDNN!");
    }

    pub unsafe fn cudnnConvolutionBackwardData(
        handle: cudnnHandle_t,
        alpha: *const c_void,
        wDesc: cudnnFilterDescriptor_t,
        w: *const c_void,
        dyDesc: cudnnTensorDescriptor_t,
        dy: *const c_void,
        convDesc: cudnnConvolutionDescriptor_t,
        algo: cudnnConvolutionBwdDataAlgo_t,
        workSpace: *mut c_void,
        workSpaceSizeInBytes: usize,
        beta: *const c_void,
        dxDesc: cudnnTensorDescriptor_t,
        dx: *mut c_void,
    ) -> cudnnStatus_t {
        unimplemented!("Convolution is not implemented without cuDNN!");
    }

    pub unsafe fn cudnnConvolutionBackwardFilter(
        handle: cudnnHandle_t,
        alpha: *const c_void,
        xDesc: cudnnTensorDescriptor_t,
        x: *const c_void,
        dyDesc: cudnnTensorDescriptor_t,
        dy: *const c_void,
        convDesc: cudnnConvolutionDescriptor_t,
        algo: cudnnConvolutionBwdFilterAlgo_t,
        workSpace: *mut c_void,
        workSpaceSizeInBytes: usize,
        beta: *const c_void,
        dwDesc: cudnnFilterDescriptor_t,
        dw: *mut c_void,
    ) -> cudnnStatus_t {
        unimplemented!("Convolution is not implemented without cuDNN!");
    }
}
