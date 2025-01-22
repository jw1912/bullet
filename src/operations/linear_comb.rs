use crate::{
    autograd::Operation,
    tensor::{DenseMatrix, ExecutionContext, Shape, Tensor},
};

#[derive(Debug)]
pub struct LinearCombination(pub f32, pub f32);

impl Operation for LinearCombination {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 2 && inputs[0] == inputs[1] {
            Ok(inputs[0])
        } else {
            Err(format!("Invalid number of inputs in add! Expected 2, got {}", inputs.len()))
        }
    }

    fn forward(&self, ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
        DenseMatrix::linear_comb(
            ctx,
            self.0,
            inputs[0].values.dense(),
            self.1,
            inputs[1].values.dense(),
            output.values.dense_mut(),
        );
    }

    fn backward(&self, ctx: &mut ExecutionContext, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        DenseMatrix::linear_comb_backward(
            ctx,
            self.0,
            input1[0].values.dense(),
            input1[0].gradients.as_mut(),
            self.1,
            input2[0].values.dense(),
            input2[0].gradients.as_mut(),
            output.gradients.as_ref().unwrap(),
        );
    }
}
