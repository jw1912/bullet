use crate::{backend::ExecutionContext, tensor::{dense_tensor::DenseTensor, Shape}, Tensor};

pub fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 2 {
        Ok(inputs[0] * inputs[1])
    } else {
        Err(String::from("Invalid number of inputs!"))
    }
}

pub fn forward(ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
    DenseTensor::matmul(
        ctx,
        &inputs[0].values,
        false,
        &inputs[1].values,
        false,
        &mut output.values,
    );
}

pub fn backprop(ctx: &mut ExecutionContext, output: &Tensor, inputs: &mut [&mut Tensor]) {
    let (input1, input2) = inputs.split_at_mut(1);

    DenseTensor::backprop_matmul(
        ctx,
        &input1[0].values,
        input1[0].gradients.as_mut(),
        false,
        &input2[0].values,
        input2[0].gradients.as_mut(),
        false,
        output.gradients.as_ref().unwrap(),
    );
}
