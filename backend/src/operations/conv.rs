use crate::backend::{dense, ConvolutionDescription, ExecutionContext, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct Convolution(pub ConvolutionDescription);

impl Operation<ExecutionContext> for Convolution {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        let conv = &self.0;

        if inputs.len() == 2 {
            if inputs[1].cols() == 1 {
                if inputs[1].size() == conv.input_shape.size() * conv.input_channels {
                    if inputs[0] == Shape::new(conv.filter_shape.size(), conv.input_channels * conv.output_channels) {
                        Ok(Shape::new(conv.output_shape.size() * conv.output_channels, 1))
                    } else {
                        Err(format!(
                            "Invalid filter size! Cannot accomodate {} {}x{} filters.",
                            conv.input_channels * conv.output_channels,
                            conv.filter_shape.rows(),
                            conv.filter_shape.cols()
                        ))
                    }
                } else {
                    Err(format!(
                        "Invalid input size! Cannot accomodate {} channels of {}x{} images.",
                        conv.input_channels,
                        conv.input_shape.rows(),
                        conv.input_shape.cols()
                    ))
                }
            } else {
                Err("Invalid input size! Convolution must take a (reshapeable) vector!".to_string())
            }
        } else {
            Err(format!("Invalid number of inputs in convolution! Expected 2, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        dense::convolution_forward(
            &self.0,
            inputs[0].values.dense(),
            inputs[1].values.dense(),
            output.values.dense_mut(),
        );
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        dense::convolution_backward(
            &self.0,
            input1[0].values.dense(),
            input1[0].gradients.as_mut(),
            input2[0].values.dense(),
            input2[0].gradients.as_mut(),
            output.gradients.as_ref().unwrap(),
        );
    }
}

#[cfg(feature = "cudnn")]
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::backend::Matrix;
    use bullet_core::device::Device;

    #[rustfmt::skip]
    #[test]
    fn conv() {
        println!("start");
        let device = Arc::new(ExecutionContext::default());

        let mut filters = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut input = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut output = Tensor::new(device.clone(), Shape::new(1, 1), true);

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

        let conv = Convolution(desc);

        assert_eq!(desc.output_shape, Shape::new(4, 4));

        filters.load_dense_from_slice(
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

        input.load_dense_from_slice(
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

        device.panic_if_device_error("Failed to load!");

        conv.forward(&[&filters, &input], &mut output);

        assert_eq!(output.shape(), Shape::new_batched(32, 1, 2));

        device.panic_if_device_error("Failed conv fwd!");

        assert_eq!(
            &output.get_dense_vals().unwrap(),
            &[
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

        if let Matrix::Dense(vals) = &output.values {
            vals.copy_into(output.gradients.as_mut().unwrap());
        }

        conv.backward(&output, &mut [&mut filters, &mut input]);

        assert_eq!(filters.gradients.as_ref().unwrap().shape(), filters.shape());
        assert_eq!(input.gradients.as_ref().unwrap().shape(), input.shape());

        let mut fbuf = vec![0.0; filters.shape().size()];
        filters.gradients.as_ref().unwrap().write_to_slice(&mut fbuf);

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

        let mut ibuf = vec![0.0; input.shape().size()];
        input.gradients.as_ref().unwrap().write_to_slice(&mut ibuf);

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
