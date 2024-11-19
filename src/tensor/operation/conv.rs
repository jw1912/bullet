use crate::{
    backend::{ConvolutionDescription, ExecutionContext},
    tensor::{DenseMatrix, Shape, Tensor},
};

pub fn output_tensor(inputs: &[Shape], desc: &ConvolutionDescription) -> Result<Shape, String> {
    if inputs.len() == 2 {
        if inputs[1].cols() == 1 {
            if inputs[1].size() == desc.input_shape.size() * desc.input_channels {
                if inputs[0] == Shape::new(desc.filter_shape.size(), desc.input_channels * desc.output_channels) {
                    Ok(Shape::new(desc.output_shape.size() * desc.output_channels, 1))
                } else {
                    Err(format!(
                        "Invalid filter size! Cannot accomodate {} {}x{} filters.",
                        desc.input_channels * desc.output_channels,
                        desc.filter_shape.rows(),
                        desc.filter_shape.cols()
                    ))
                }
            } else {
                Err(format!(
                    "Invalid input size! Cannot accomodate {} channels of {}x{} images.",
                    desc.input_channels,
                    desc.input_shape.rows(),
                    desc.input_shape.cols()
                ))
            }
        } else {
            Err("Invalid input size! Convolution must take a (reshapeable) vector!".to_string())
        }
    } else {
        Err(format!("Invalid number of inputs in convolution! Expected 2, got {}", inputs.len()))
    }
}

pub fn forward(ctx: &mut ExecutionContext, desc: &ConvolutionDescription, inputs: &[&Tensor], output: &mut Tensor) {
    DenseMatrix::convolution_forward(
        ctx,
        desc,
        inputs[0].values.dense(),
        inputs[1].values.dense(),
        output.values.dense_mut(),
    );
}

pub fn backprop(
    ctx: &mut ExecutionContext,
    desc: &ConvolutionDescription,
    output: &Tensor,
    inputs: &mut [&mut Tensor],
) {
    let (input1, input2) = inputs.split_at_mut(1);

    DenseMatrix::convolution_backward(
        ctx,
        desc,
        input1[0].values.dense(),
        input1[0].gradients.as_mut(),
        input2[0].values.dense(),
        input2[0].gradients.as_mut(),
        output.gradients.as_ref().unwrap(),
    );
}
