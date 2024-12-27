use crate::{
    backend::ExecutionContext,
    tensor::{DenseMatrix, Shape, Tensor},
};

pub fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 2 && inputs[0] == inputs[1] {
        Ok(inputs[0])
    } else {
        Err(format!("Invalid number of inputs in add! Expected 2, got {}", inputs.len()))
    }
}

pub fn forward(ctx: &mut ExecutionContext, alpha: f32, beta: f32, inputs: &[&Tensor], output: &mut Tensor) {
    DenseMatrix::linear_comb(
        ctx,
        alpha,
        inputs[0].values.dense(),
        beta,
        inputs[1].values.dense(),
        output.values.dense_mut(),
    );
}

pub fn backprop(ctx: &mut ExecutionContext, alpha: f32, beta: f32, output: &Tensor, inputs: &mut [&mut Tensor]) {
    let (input1, input2) = inputs.split_at_mut(1);

    DenseMatrix::linear_comb_backward(
        ctx,
        alpha,
        input1[0].values.dense(),
        input1[0].gradients.as_mut(),
        beta,
        input2[0].values.dense(),
        input2[0].gradients.as_mut(),
        output.gradients.as_ref().unwrap(),
    );
}
