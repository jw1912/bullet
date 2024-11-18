use crate::{
    backend::{ConvolutionDescription, ExecutionContext},
    tensor::{DenseMatrix, Shape, Tensor},
};

pub fn output_tensor(inputs: &[Shape], desc: &ConvolutionDescription) -> Result<Shape, String> {
    if inputs.len() == 2 {
        if inputs[1].cols() == 1 {
            if inputs[1].size() == desc.input_shape.size() * desc.input_channels {
                if inputs[0] == Shape::new(desc.filter_shape.size(), desc.input_channels * desc.output_channels) {
                    Ok(inputs[0])
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
    _ctx: &mut ExecutionContext,
    _desc: &ConvolutionDescription,
    _output: &Tensor,
    _inputs: &mut [&mut Tensor],
) {
    unimplemented!("Not yet implemented!");
}
