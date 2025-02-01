use crate::backend::{sparse, ExecutionContext, Matrix, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct Select;

impl Operation<ExecutionContext> for Select {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 2 {
            if inputs[0].cols() == inputs[1].cols() && inputs[0].rows() % inputs[1].rows() == 0 {
                Ok(Shape::new(inputs[0].rows() / inputs[1].rows(), inputs[0].cols()))
            } else {
                Err(String::from("Vector cannot be split evenly among buckets!"))
            }
        } else {
            Err(format!("Invalid number of inputs in select! Expected 2, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        if let Matrix::Sparse(buckets) = &inputs[1].values {
            sparse::select(inputs[0].values.dense(), buckets, output.values.dense_mut());
        } else {
            panic!("Bucket indices must be integers!")
        }
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);
        let output_grad = output.gradients.as_ref().unwrap();

        if let Some(grad) = input1[0].gradients.as_mut() {
            if let Matrix::Sparse(buckets) = &input2[0].values {
                sparse::select_backprop(input1[0].values.dense(), buckets, output_grad, grad);
            } else {
                panic!("Bucket indices must be integers!")
            }
        }
    }
}
