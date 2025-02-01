use crate::backend::{dense, sparse, ExecutionContext, Matrix, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct Linear;

impl Operation<ExecutionContext> for Linear {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 2 {
            Ok(inputs[0] * inputs[1])
        } else {
            Err(format!("Invalid number of inputs in linear! Expected 2, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        match &inputs[1].values {
            Matrix::Dense(dense) => {
                dense::matmul(inputs[0].values.dense(), false, dense, false, output.values.dense_mut());
            }
            Matrix::Sparse(sparse) => {
                sparse::linear(inputs[0].values.dense(), sparse, output.values.dense_mut());
            }
        }
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        match &input2[0].values {
            Matrix::Dense(dense) => {
                dense::backprop_matmul(
                    input1[0].values.dense(),
                    input1[0].gradients.as_mut(),
                    false,
                    dense,
                    input2[0].gradients.as_mut(),
                    false,
                    output.gradients.as_ref().unwrap(),
                );
            }
            Matrix::Sparse(sparse) => {
                assert!(input2[0].gradients.as_ref().is_none());

                if let Some(grad) = input1[0].gradients.as_mut() {
                    sparse::backprop_linear(
                        input1[0].values.dense(),
                        grad,
                        sparse,
                        output.values.dense(),
                        output.gradients.as_ref().unwrap(),
                    );
                }
            }
        }
    }
}
