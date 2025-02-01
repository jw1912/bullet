use crate::{
    autograd::Operation,
    tensor::{DenseMatrix, ExecutionContext, Shape, Tensor},
};

#[derive(Debug)]
pub struct SliceRows(pub usize, pub usize);

impl Operation for SliceRows {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 1 {
            if self.1 > self.0 {
                if self.1 <= inputs[0].rows() {
                    Ok(Shape::new(self.1 - self.0, inputs[0].cols()))
                } else {
                    Err(format!("Invalid slice indices! end = {} > rows = {}", self.1, inputs[0].rows()))
                }
            } else {
                Err(format!("Invalid slice indices! start = {} >= end = {}", self.0, self.1))
            }
        } else {
            Err(format!("Invalid number of inputs in slice! Expected 1, got {}", inputs.len()))
        }
    }

    fn forward(&self, ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
        DenseMatrix::slice_rows(ctx, inputs[0].values.dense(), self.0, self.1, output.values.dense_mut());
    }

    fn backward(&self, ctx: &mut ExecutionContext, output: &Tensor, inputs: &mut [&mut Tensor]) {
        DenseMatrix::backprop_slice_rows(
            ctx,
            inputs[0].values.dense(),
            inputs[0].gradients.as_mut(),
            self.0,
            self.1,
            output.gradients.as_ref().unwrap(),
        );
    }
}
