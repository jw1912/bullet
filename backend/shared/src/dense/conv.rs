use bullet_core::device::DeviceBuffer;

use crate::{
    backend::conv::{self, ConvolutionCudnnDescription, ConvolutionDescription},
    DenseMatrix, Shape,
};

pub fn convolution_forward(
    desc: &ConvolutionDescription,
    filters: &DenseMatrix,
    input: &DenseMatrix,
    output: &mut DenseMatrix,
) {
    assert!(filters.shape.batch_size().is_none());
    assert_eq!(filters.shape.rows(), desc.filter_shape.size());
    assert_eq!(filters.shape.cols(), desc.input_channels * desc.output_channels);
    assert_eq!(input.shape.rows(), desc.input_shape.size() * desc.input_channels);

    let cudnn_desc = ConvolutionCudnnDescription::new(desc, input.shape.batch_size().unwrap_or(1));

    output.reshape_if_needed(Shape::from_raw(
        desc.output_shape.size() * desc.output_channels,
        1,
        input.shape.batch_size(),
    ));

    unsafe {
        conv::conv_fwd(
            input.buf.device().as_ref(),
            &cudnn_desc,
            input.buf.ptr(),
            filters.buf.ptr(),
            output.buf.mut_ptr(),
        );
    }
}

pub fn convolution_backward(
    desc: &ConvolutionDescription,
    filters: &DenseMatrix,
    filters_grad: Option<&mut DenseMatrix>,
    input: &DenseMatrix,
    input_grad: Option<&mut DenseMatrix>,
    output_grad: &DenseMatrix,
) {
    assert!(filters.shape.batch_size().is_none());

    assert_eq!(filters.shape.rows(), desc.filter_shape.size());
    assert_eq!(filters.shape.cols(), desc.input_channels * desc.output_channels);
    assert_eq!(input.shape.rows(), desc.input_shape.size() * desc.input_channels);
    assert_eq!(output_grad.shape.rows(), desc.output_shape.size() * desc.output_channels);
    assert_eq!(output_grad.shape.cols(), input.shape.cols());

    let cudnn_desc = ConvolutionCudnnDescription::new(desc, input.shape.batch_size().unwrap_or(1));

    let device = filters.buf.device();
    let ctx = device.as_ref();

    if let Some(grad) = filters_grad {
        grad.reshape_if_needed(filters.shape);

        unsafe {
            conv::conv_bwd_filter(ctx, &cudnn_desc, input.buf.ptr(), output_grad.buf.ptr(), grad.buf.mut_ptr());
        }
    }

    if let Some(grad) = input_grad {
        grad.reshape_if_needed(input.shape);

        unsafe {
            conv::conv_bwd_data(ctx, &cudnn_desc, filters.buf.ptr(), output_grad.buf.ptr(), grad.buf.mut_ptr());
        }
    }
}
