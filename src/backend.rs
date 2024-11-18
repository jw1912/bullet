pub mod bindings;
mod buffer;
pub mod ops;
pub mod util;

use std::ffi::c_int;

use bindings::{
    cublasHandle_t, cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, cudnnConvolutionMode_t,
    cudnnCreateConvolutionDescriptor, cudnnCreateFilterDescriptor, cudnnCreateTensorDescriptor, cudnnDataType_t,
    cudnnDestroyConvolutionDescriptor, cudnnDestroyFilterDescriptor, cudnnDestroyTensorDescriptor,
    cudnnFilterDescriptor_t, cudnnHandle_t, cudnnSetConvolution2dDescriptor, cudnnSetFilter4dDescriptor,
    cudnnSetTensor4dDescriptor, cudnnStatus_t, cudnnTensorDescriptor_t, cudnnTensorFormat_t,
};
pub use buffer::Buffer;

use crate::Shape;

/// This contains the internal environment for the GPU to use:
/// - BLAS handles
/// - Internal buffers for use in operations without additional allocation overhead
#[derive(Debug)]
pub struct ExecutionContext {
    handle: cublasHandle_t,
    pub(crate) cudnn: cudnnHandle_t,
    ones: Buffer<f32>,
}

impl Drop for ExecutionContext {
    fn drop(&mut self) {
        unsafe {
            let status = bindings::cublasDestroy_v2(self.handle);
            assert_eq!(status, bindings::CUBLAS_SUCCESS);

            let status = bindings::cudnnDestroy(self.cudnn);
            assert_eq!(status, bindings::cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        }
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        let mut handle: cublasHandle_t = std::ptr::null_mut();
        let mut cudnn: cudnnHandle_t = std::ptr::null_mut();

        unsafe {
            let status = bindings::cublasCreate_v2((&mut handle) as *mut cublasHandle_t);
            assert_eq!(status, bindings::CUBLAS_SUCCESS);

            let status = bindings::cudnnCreate((&mut cudnn) as *mut cudnnHandle_t);
            assert_eq!(status, bindings::cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        }

        let ones = Buffer::new(1);
        ones.load_from_slice(&[1.0]);

        Self { handle, cudnn, ones }
    }
}

pub struct ConvolutionDescription {
    pub input_shape: Shape,
    pub input_channels: usize,
    pub output_shape: Shape,
    pub output_channels: usize,
    pub filter_shape: Shape,
    pub padding_shape: Shape,
    pub stride_shape: Shape,
    pub input: cudnnTensorDescriptor_t,
    pub filter: cudnnFilterDescriptor_t,
    pub conv: cudnnConvolutionDescriptor_t,
    pub algo: cudnnConvolutionFwdAlgo_t,
    pub output: cudnnTensorDescriptor_t,
}

pub fn catch_cudnn(status: cudnnStatus_t) {
    if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        panic!("cuDNN error: {status:?}");
    }
}

impl Drop for ConvolutionDescription {
    fn drop(&mut self) {
        unsafe {
            catch_cudnn(cudnnDestroyTensorDescriptor(self.input));
            catch_cudnn(cudnnDestroyFilterDescriptor(self.filter));
            catch_cudnn(cudnnDestroyConvolutionDescriptor(self.conv));
            catch_cudnn(cudnnDestroyTensorDescriptor(self.output));
        }
    }
}

impl ConvolutionDescription {
    pub fn new(
        input_shape: Shape,
        input_channels: usize,
        output_channels: usize,
        filter_shape: Shape,
        padding_shape: Shape,
        stride_shape: Shape,
    ) -> Self {
        let hout = (input_shape.rows() + 2 * padding_shape.rows() - filter_shape.rows()) / stride_shape.rows() + 1;
        let wout = (input_shape.cols() + 2 * padding_shape.cols() - filter_shape.cols()) / stride_shape.cols() + 1;

        let mut desc = Self {
            input_shape,
            input_channels,
            output_shape: Shape::new(hout, wout),
            output_channels,
            filter_shape,
            padding_shape,
            stride_shape,
            input: std::ptr::null_mut(),
            filter: std::ptr::null_mut(),
            conv: std::ptr::null_mut(),
            algo: cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            output: std::ptr::null_mut(),
        };

        unsafe {
            catch_cudnn(cudnnCreateTensorDescriptor((&mut desc.input) as *mut cudnnTensorDescriptor_t));
            catch_cudnn(cudnnCreateFilterDescriptor((&mut desc.filter) as *mut cudnnFilterDescriptor_t));
            catch_cudnn(cudnnCreateConvolutionDescriptor((&mut desc.conv) as *mut cudnnConvolutionDescriptor_t));
            catch_cudnn(cudnnCreateTensorDescriptor((&mut desc.output) as *mut cudnnTensorDescriptor_t));

            desc.set_descriptors(1);
        }

        desc
    }

    pub fn set_descriptors(&self, batch_size: usize) {
        unsafe {
            catch_cudnn(cudnnSetTensor4dDescriptor(
                self.input,
                cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                batch_size as c_int,
                self.input_channels as c_int,
                self.input_shape.rows() as c_int,
                self.input_shape.cols() as c_int,
            ));

            catch_cudnn(cudnnSetFilter4dDescriptor(
                self.filter,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                self.output_channels as c_int,
                self.input_channels as c_int,
                self.filter_shape.rows() as c_int,
                self.filter_shape.cols() as c_int,
            ));

            catch_cudnn(cudnnSetConvolution2dDescriptor(
                self.conv,
                self.padding_shape.rows() as c_int,
                self.padding_shape.cols() as c_int,
                self.stride_shape.rows() as c_int,
                self.stride_shape.cols() as c_int,
                1,
                1,
                cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
            ));

            catch_cudnn(cudnnSetTensor4dDescriptor(
                self.output,
                cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                batch_size as c_int,
                self.output_channels as c_int,
                self.output_shape.rows() as c_int,
                self.output_shape.cols() as c_int,
            ));
        }
    }
}
