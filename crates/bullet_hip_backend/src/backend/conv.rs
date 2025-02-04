use std::ffi::c_int;

use super::{
    bindings::{
        cudnnConvolutionBackwardData, cudnnConvolutionBackwardFilter, cudnnConvolutionBwdDataAlgo_t,
        cudnnConvolutionBwdFilterAlgo_t, cudnnConvolutionDescriptor_t, cudnnConvolutionForward,
        cudnnConvolutionFwdAlgo_t, cudnnConvolutionMode_t, cudnnCreateConvolutionDescriptor,
        cudnnCreateFilterDescriptor, cudnnCreateTensorDescriptor, cudnnDataType_t, cudnnDestroyConvolutionDescriptor,
        cudnnDestroyFilterDescriptor, cudnnDestroyTensorDescriptor, cudnnFilterDescriptor_t,
        cudnnSetConvolution2dDescriptor, cudnnSetFilter4dDescriptor, cudnnSetTensor4dDescriptor, cudnnStatus_t,
        cudnnTensorDescriptor_t, cudnnTensorFormat_t,
    },
    ExecutionContext,
};

use bullet_core::graph::operation::ConvolutionDescription;

pub unsafe fn conv_fwd(
    ctx: &ExecutionContext,
    desc: &ConvolutionCudnnDescription,
    input: *const f32,
    filters: *const f32,
    output: *mut f32,
) {
    let alpha = 1f32;
    let beta = 0f32;

    catch_cudnn(cudnnConvolutionForward(
        ctx.cudnn,
        ((&alpha) as *const f32).cast(),
        desc.input,
        input.cast(),
        desc.filter,
        filters.cast(),
        desc.conv,
        desc.fwd_algo,
        std::ptr::null_mut(),
        0,
        ((&beta) as *const f32).cast(),
        desc.output,
        output.cast(),
    ));
}

pub unsafe fn conv_bwd_filter(
    ctx: &ExecutionContext,
    desc: &ConvolutionCudnnDescription,
    input: *const f32,
    output_grad: *const f32,
    input_grad: *mut f32,
) {
    let alpha = 1f32;
    let beta = 0f32;

    catch_cudnn(cudnnConvolutionBackwardFilter(
        ctx.cudnn,
        ((&alpha) as *const f32).cast(),
        desc.input,
        input.cast(),
        desc.output,
        output_grad.cast(),
        desc.conv,
        desc.bwd_filter_algo,
        std::ptr::null_mut(),
        0,
        ((&beta) as *const f32).cast(),
        desc.filter,
        input_grad.cast(),
    ));
}

pub unsafe fn conv_bwd_data(
    ctx: &ExecutionContext,
    desc: &ConvolutionCudnnDescription,
    filters: *const f32,
    output_grad: *const f32,
    input_grad: *mut f32,
) {
    let alpha = 1f32;
    let beta = 0f32;

    catch_cudnn(cudnnConvolutionBackwardData(
        ctx.cudnn,
        ((&alpha) as *const f32).cast(),
        desc.filter,
        filters.cast(),
        desc.output,
        output_grad.cast(),
        desc.conv,
        desc.bwd_data_algo,
        std::ptr::null_mut(),
        0,
        ((&beta) as *const f32).cast(),
        desc.input,
        input_grad.cast(),
    ));
}

fn catch_cudnn(status: cudnnStatus_t) {
    if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        panic!("cuDNN error: {status:?}");
    }
}

pub struct ConvolutionCudnnDescription {
    pub input: cudnnTensorDescriptor_t,
    pub filter: cudnnFilterDescriptor_t,
    pub conv: cudnnConvolutionDescriptor_t,
    pub fwd_algo: cudnnConvolutionFwdAlgo_t,
    pub bwd_data_algo: cudnnConvolutionBwdDataAlgo_t,
    pub bwd_filter_algo: cudnnConvolutionBwdFilterAlgo_t,
    pub output: cudnnTensorDescriptor_t,
}

impl Drop for ConvolutionCudnnDescription {
    fn drop(&mut self) {
        unsafe {
            catch_cudnn(cudnnDestroyTensorDescriptor(self.input));
            catch_cudnn(cudnnDestroyFilterDescriptor(self.filter));
            catch_cudnn(cudnnDestroyConvolutionDescriptor(self.conv));
            catch_cudnn(cudnnDestroyTensorDescriptor(self.output));
        }
    }
}

impl ConvolutionCudnnDescription {
    pub fn new(desc: &ConvolutionDescription, batch_size: usize) -> Self {
        let mut res = Self {
            input: std::ptr::null_mut(),
            filter: std::ptr::null_mut(),
            conv: std::ptr::null_mut(),
            fwd_algo: cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            bwd_data_algo: cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            bwd_filter_algo: cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            output: std::ptr::null_mut(),
        };

        unsafe {
            catch_cudnn(cudnnCreateTensorDescriptor((&mut res.input) as *mut cudnnTensorDescriptor_t));
            catch_cudnn(cudnnCreateFilterDescriptor((&mut res.filter) as *mut cudnnFilterDescriptor_t));
            catch_cudnn(cudnnCreateConvolutionDescriptor((&mut res.conv) as *mut cudnnConvolutionDescriptor_t));
            catch_cudnn(cudnnCreateTensorDescriptor((&mut res.output) as *mut cudnnTensorDescriptor_t));

            catch_cudnn(cudnnSetTensor4dDescriptor(
                res.input,
                cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                batch_size as c_int,
                desc.input_channels as c_int,
                desc.input_shape.rows() as c_int,
                desc.input_shape.cols() as c_int,
            ));

            catch_cudnn(cudnnSetFilter4dDescriptor(
                res.filter,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                desc.output_channels as c_int,
                desc.input_channels as c_int,
                desc.filter_shape.rows() as c_int,
                desc.filter_shape.cols() as c_int,
            ));

            catch_cudnn(cudnnSetConvolution2dDescriptor(
                res.conv,
                desc.padding_shape.0 as c_int,
                desc.padding_shape.1 as c_int,
                desc.stride_shape.rows() as c_int,
                desc.stride_shape.cols() as c_int,
                1,
                1,
                cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
            ));

            catch_cudnn(cudnnSetTensor4dDescriptor(
                res.output,
                cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                batch_size as c_int,
                desc.output_channels as c_int,
                desc.output_shape.rows() as c_int,
                desc.output_shape.cols() as c_int,
            ));
        }

        res
    }
}
