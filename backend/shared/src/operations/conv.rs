use crate::{
    autograd::Operation,
    tensor::{ConvolutionDescription, DenseMatrix, ExecutionContext, Shape, Tensor},
};

impl Operation for ConvolutionDescription {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 2 {
            if inputs[1].cols() == 1 {
                if inputs[1].size() == self.input_shape.size() * self.input_channels {
                    if inputs[0] == Shape::new(self.filter_shape.size(), self.input_channels * self.output_channels) {
                        Ok(Shape::new(self.output_shape.size() * self.output_channels, 1))
                    } else {
                        Err(format!(
                            "Invalid filter size! Cannot accomodate {} {}x{} filters.",
                            self.input_channels * self.output_channels,
                            self.filter_shape.rows(),
                            self.filter_shape.cols()
                        ))
                    }
                } else {
                    Err(format!(
                        "Invalid input size! Cannot accomodate {} channels of {}x{} images.",
                        self.input_channels,
                        self.input_shape.rows(),
                        self.input_shape.cols()
                    ))
                }
            } else {
                Err("Invalid input size! Convolution must take a (reshapeable) vector!".to_string())
            }
        } else {
            Err(format!("Invalid number of inputs in convolution! Expected 2, got {}", inputs.len()))
        }
    }

    fn forward(&self, ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
        DenseMatrix::convolution_forward(
            ctx,
            self,
            inputs[0].values.dense(),
            inputs[1].values.dense(),
            output.values.dense_mut(),
        );
    }

    fn backward(&self, ctx: &mut ExecutionContext, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        DenseMatrix::convolution_backward(
            ctx,
            self,
            input1[0].values.dense(),
            input1[0].gradients.as_mut(),
            input2[0].values.dense(),
            input2[0].gradients.as_mut(),
            output.gradients.as_ref().unwrap(),
        );
    }
}
