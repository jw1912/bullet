use diffable::DiffableOperation;

use crate::{backend::ExecutionContext, tensor::{raw_tensor::RawTensor, Shape}, Tensor};

pub fn linear() -> DiffableOperation<Tensor> {
    DiffableOperation {
        output_tensor,
        forward,
        backprop,
    }
}

fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 2 {
        Ok(inputs[0] * inputs[1])
    } else {
        Err(String::from("Invalid number of inputs!"))
    }
}

fn forward(ctx: &ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
    RawTensor::linear_transform_forward_batched(
        ctx,
        &inputs[0].values,
        &inputs[1].values,
        &mut output.values,
    );
}

fn backprop(ctx: &ExecutionContext, output: &Tensor, inputs: &mut [&mut Tensor]) {
    RawTensor::linear_transform_backward_batched(
        ctx,
        inputs,
        output.gradients.as_ref().unwrap(),
    );
}
