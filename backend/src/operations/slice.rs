use crate::backend::{dense, ExecutionContext, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct SliceRows(pub usize, pub usize);

impl Operation<ExecutionContext> for SliceRows {
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

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        dense::slice_rows(inputs[0].values.dense(), self.0, self.1, output.values.dense_mut());
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        dense::backprop_slice_rows(
            inputs[0].values.dense(),
            inputs[0].gradients.as_mut(),
            self.0,
            self.1,
            output.gradients.as_ref().unwrap(),
        );
    }
}
