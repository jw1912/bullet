use crate::backend::{dense, ExecutionContext, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct PairwiseMul(pub bool);

impl Operation<ExecutionContext> for PairwiseMul {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 1 {
            let input = inputs[0];
            if input.rows() % 2 != 0 {
                Err(String::from("Input size must be even!"))
            } else {
                Ok(Shape::new(input.rows() / 2, input.cols()))
            }
        } else {
            Err(format!("Invalid number of inputs in pairwise! Expected 1, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        dense::pairwise(inputs[0].values.dense(), output.values.dense_mut(), self.0);
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let input = inputs[0].values.dense();
        let output_grad = output.gradients.as_ref().expect("Must exist!");
        if let Some(input_grad) = inputs[0].gradients.as_mut() {
            dense::backprop_pairwise(input, output_grad, input_grad, self.0);
        }
    }
}
