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

#[cfg(feature = "cudnn")]
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{backend::util::panic_if_device_error, ExecutionContext, Shape};

    use super::*;

    #[rustfmt::skip]
    #[test]
    fn conv() {
        println!("start");
        let device = Arc::new(ExecutionContext::default());

        let mut filters = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        let input_shape = Shape::new(4, 4);
        let input_channels = 1;
        let output_channels = 2;
        let filter_shape = Shape::new(3, 3);

        let desc = ConvolutionDescription::new(
            input_shape,
            input_channels,
            output_channels,
            filter_shape,
            (1, 1),
            Shape::new(1, 1),
        );

        assert_eq!(desc.output_shape, Shape::new(4, 4));

        filters.load_from_slice(
            Shape::new(filter_shape.size(), input_channels * output_channels),
            &[
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,

                -1.0, -1.0, -1.0,
                -1.0, -1.0, -1.0,
                -1.0, -1.0, -1.0,
            ],
        );

        input.load_from_slice(
            Shape::new_batched(16, 1, 2),
            &[
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,

                -1.0, -2.0, -3.0, -4.0,
                -5.0, -6.0, -7.0, -8.0,
                -1.0, -2.0, -3.0, -4.0,
                -5.0, -6.0, -7.0, -8.0,
            ],
        );

        panic_if_device_error("Failed to load!");

        convolution_forward(
            &desc,
            &filters,
            &input,
            &mut output,
        );

        assert_eq!(output.shape(), Shape::new_batched(32, 1, 2));

        panic_if_device_error("Failed conv fwd!");

        let mut buf = [0.0; 64];
        output.write_to_slice(&mut buf);

        assert_eq!(
            buf,
            [
                14.0, 24.0, 30.0, 22.0,
                17.0, 30.0, 39.0, 29.0,
                25.0, 42.0, 51.0, 37.0,
                14.0, 24.0, 30.0, 22.0,

                -14.0, -24.0, -30.0, -22.0,
                -17.0, -30.0, -39.0, -29.0,
                -25.0, -42.0, -51.0, -37.0,
                -14.0, -24.0, -30.0, -22.0,

                -14.0, -24.0, -30.0, -22.0,
                -17.0, -30.0, -39.0, -29.0,
                -25.0, -42.0, -51.0, -37.0,
                -14.0, -24.0, -30.0, -22.0,

                14.0, 24.0, 30.0, 22.0,
                17.0, 30.0, 39.0, 29.0,
                25.0, 42.0, 51.0, 37.0,
                14.0, 24.0, 30.0, 22.0,
            ],
        );

        let mut filters_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        convolution_backward(
            &desc,
            &filters,
            Some(&mut filters_grad),
            &input,
            Some(&mut input_grad),
            &output,
        );

        assert_eq!(filters_grad.shape, filters.shape);
        assert_eq!(input_grad.shape, input.shape);

        let mut fbuf = vec![0.0; filters.shape.size()];
        filters_grad.write_to_slice(&mut fbuf);

        assert_eq!(
            &fbuf,
            &[
                2240.0, 3160.0, 2704.0,
                2892.0, 4040.0, 3432.0,
                2848.0, 3880.0, 3248.0,

                -2240.0, -3160.0, -2704.0,
                -2892.0, -4040.0, -3432.0,
                -2848.0, -3880.0, -3248.0,
            ],
        );

        let mut ibuf = vec![0.0; input.shape.size()];
        input_grad.write_to_slice(&mut ibuf);

        assert_eq!(
            &ibuf,
            &[
                170.0, 308.0, 348.0, 240.0,
                304.0, 544.0, 608.0, 416.0,
                304.0, 544.0, 608.0, 416.0,
                210.0, 372.0, 412.0, 280.0,

                -170.0, -308.0, -348.0, -240.0,
                -304.0, -544.0, -608.0, -416.0,
                -304.0, -544.0, -608.0, -416.0,
                -210.0, -372.0, -412.0, -280.0,
            ],
        );
    }
}
