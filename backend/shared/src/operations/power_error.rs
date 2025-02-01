use crate::{
    autograd::Operation,
    tensor::{DenseMatrix, ExecutionContext, Shape, Tensor},
};

#[derive(Debug)]
pub struct AbsPowerError(pub f32);

impl Operation for AbsPowerError {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 2 && inputs[0] == inputs[1] {
            Ok(Shape::new(1, 1))
        } else {
            Err(format!("Invalid number of inputs in power error! Expected 1, got {}", inputs.len()))
        }
    }

    fn forward(&self, _: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
        DenseMatrix::abs_power_error(
            self.0,
            inputs[0].values.dense(),
            inputs[1].values.dense(),
            output.values.dense_mut(),
        );
    }

    fn backward(&self, _: &mut ExecutionContext, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        DenseMatrix::backprop_abs_power_error(
            self.0,
            input1[0].values.dense(),
            input1[0].gradients.as_mut(),
            input2[0].values.dense(),
            input2[0].gradients.as_mut(),
            output.gradients.as_ref().unwrap(),
        );
    }
}
