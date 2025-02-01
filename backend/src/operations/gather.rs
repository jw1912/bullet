use crate::backend::{sparse, ExecutionContext, Matrix, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct Gather;

impl Operation<ExecutionContext> for Gather {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 2 {
            if inputs[0].cols() == 1 && inputs[1].cols() == 1 {
                Ok(inputs[1])
            } else {
                Err("Both inputs must be vectors!".to_string())
            }
        } else {
            Err(format!("Invalid number of inputs in linear! Expected 2, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        match &inputs[1].values {
            Matrix::Dense(_) => unimplemented!("Masking with dense masks is not supported!"),
            Matrix::Sparse(sparse) => {
                sparse::gather(inputs[0].values.dense(), sparse, output.values.dense_mut());
            }
        }
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        match &input2[0].values {
            Matrix::Dense(_) => unimplemented!("Masking with dense masks is not supported!"),
            Matrix::Sparse(sparse) => {
                assert!(input2[0].gradients.as_ref().is_none());

                if let Some(grad) = input1[0].gradients.as_mut() {
                    sparse::backprop_gather(output.gradients.as_ref().unwrap(), sparse, input1[0].values.dense(), grad);
                }
            }
        }
    }
}
