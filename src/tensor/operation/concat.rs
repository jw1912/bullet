use crate::{
    backend::ExecutionContext,
    tensor::{DenseMatrix, Shape, Tensor},
};

pub fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 2 && inputs[0].cols() == inputs[1].cols() {
        Ok(Shape::new(inputs[0].rows() + inputs[1].rows(), inputs[0].cols()))
    } else {
        Err(format!("Invalid number of inputs in concat! Expected 2, got {}", inputs.len()))
    }
}

pub fn forward(ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
    DenseMatrix::concat(ctx, inputs[0].values.dense(), inputs[1].values.dense(), output.values.dense_mut());
}

pub fn backprop(ctx: &mut ExecutionContext, output: &Tensor, inputs: &mut [&mut Tensor]) {
    let (input1, input2) = inputs.split_at_mut(1);

    DenseMatrix::backprop_concat(
        ctx,
        input1[0].values.dense(),
        input1[0].gradients.as_mut(),
        input2[0].values.dense(),
        input2[0].gradients.as_mut(),
        output.gradients.as_ref().unwrap(),
    );
}
