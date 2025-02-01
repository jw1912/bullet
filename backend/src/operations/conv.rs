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
