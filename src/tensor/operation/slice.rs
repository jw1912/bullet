use crate::{
    backend::ExecutionContext,
    tensor::{DenseMatrix, Shape, Tensor},
};

pub fn output_tensor(inputs: &[Shape], start: usize, end: usize) -> Result<Shape, String> {
    if inputs.len() == 1 {
        if end > start {
            if end <= inputs[0].rows() {
                Ok(Shape::new(end - start, inputs[0].cols()))
            } else {
                Err(format!("Invalid slice indices! end = {end} > rows = {}", inputs[0].rows()))
            }
        } else {
            Err(format!("Invalid slice indices! start = {start} >= end = {end}"))
        }
    } else {
        Err(format!("Invalid number of inputs in slice! Expected 1, got {}", inputs.len()))
    }
}

pub fn forward(ctx: &mut ExecutionContext, inputs: &[&Tensor], start: usize, end: usize, output: &mut Tensor) {
    DenseMatrix::slice_rows(ctx, inputs[0].values.dense(), start, end, output.values.dense_mut());
}

pub fn backprop(ctx: &mut ExecutionContext, output: &Tensor, start: usize, end: usize, inputs: &mut [&mut Tensor]) {
    DenseMatrix::backprop_slice_rows(
        ctx,
        inputs[0].values.dense(),
        inputs[0].gradients.as_mut(),
        start,
        end,
        output.gradients.as_ref().unwrap(),
    );
}
