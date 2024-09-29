use crate::{
    backend::ExecutionContext,
    tensor::{dense_matrix::DenseMatrix, Shape},
    Tensor,
};

pub fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 2 && inputs[0] == inputs[1] {
        Ok(inputs[0])
    } else {
        Err(String::from("Invalid number of inputs!"))
    }
}

pub fn forward(ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
    DenseMatrix::add(ctx, &inputs[0].values, &inputs[1].values, &mut output.values);
}

pub fn backprop(ctx: &mut ExecutionContext, output: &Tensor, inputs: &mut [&mut Tensor]) {
    let (input1, input2) = inputs.split_at_mut(1);

    DenseMatrix::add_backward(
        ctx,
        &input1[0].values,
        input1[0].gradients.as_mut(),
        &input2[0].values,
        input2[0].gradients.as_mut(),
        output.gradients.as_ref().unwrap(),
    );
}
