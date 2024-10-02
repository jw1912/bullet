use crate::{
    backend::ExecutionContext,
    tensor::{DenseMatrix, Shape},
    Tensor,
};

pub fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 3 && inputs[0] * inputs[1] == inputs[2] {
        Ok(inputs[2])
    } else {
        Err(String::from("Invalid number of inputs!"))
    }
}

pub fn forward(ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
    super::linear::forward(ctx, &[inputs[0], inputs[1]], output);
    DenseMatrix::add_assign_vector_to_matrix_columns(ctx, inputs[2].values.dense(), output.values.dense_mut());
}

pub fn backprop(ctx: &mut ExecutionContext, output: &Tensor, inputs: &mut [&mut Tensor]) {
    let (input1, inputs2) = inputs.split_at_mut(1);
    let (input2, input3) = inputs2.split_at_mut(1);

    super::linear::backprop(ctx, output, &mut [&mut input1[0], &mut input2[0]]);

    if let Some(grad) = &mut input3[0].gradients {
        DenseMatrix::backprop_add_single(ctx, input3[0].values.dense(), grad, output.gradients.as_ref().unwrap());
    }
}
