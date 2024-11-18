use crate::{backend::{bindings, catch_cudnn, ConvolutionDescription, ExecutionContext}, Shape};

use super::DenseMatrix;

impl DenseMatrix {
    pub fn convolution_forward(
        ctx: &mut ExecutionContext,
        desc: &ConvolutionDescription,
        filters: &DenseMatrix,
        input: &DenseMatrix,
        output: &mut DenseMatrix,
    ) {
        assert_eq!(filters.shape.rows(), desc.filter_shape.size());
        assert_eq!(filters.shape.cols(), desc.input_channels * desc.output_channels);
        assert_eq!(input.shape.rows(), desc.input_shape.size() * desc.input_channels);
        desc.set_descriptors(input.shape.cols());

        output.reshape_if_needed(Shape::new(desc.output_shape.size() * desc.output_channels, input.shape.cols()));

        let alpha = 1f32;
        let beta = 0f32;

        unsafe {
            catch_cudnn(
                bindings::cudnnConvolutionForward(
                    ctx.cudnn,
                    ((&alpha) as *const f32).cast(),
                    desc.input,
                    input.buf.ptr().cast(),
                    desc.filter,
                    filters.buf.ptr().cast(),
                    desc.conv,
                    desc.algo,
                    std::ptr::null_mut(),
                    0,
                    ((&beta) as *const f32).cast(),
                    desc.output,
                    output.buf.mut_ptr().cast(),
                )
            );
        }
    }
}

#[cfg(feature = "cudnn")]
#[cfg(test)]
mod tests {
    use crate::{backend::util::panic_if_device_error, Shape};

    use super::*;

    #[rustfmt::skip]
    #[test]
    fn conv_fwd() {
        println!("start");
        let mut ctx = ExecutionContext::default();

        let mut filters = DenseMatrix::default();
        let mut input = DenseMatrix::default();
        let mut output = DenseMatrix::default();

        let input_shape = Shape::new(4, 4);
        let input_channels = 1;
        let output_channels = 2;
        let filter_shape = Shape::new(3, 3);

        let desc = ConvolutionDescription::new(
            input_shape,
            input_channels,
            output_channels,
            filter_shape,
            Shape::new(1, 1),
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
            Shape::new(16, 2),
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

        DenseMatrix::convolution_forward(
            &mut ctx,
            &desc,
            &filters,
            &input,
            &mut output,
        );

        assert_eq!(output.shape(), Shape::new(32, 2));

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
    }
}
