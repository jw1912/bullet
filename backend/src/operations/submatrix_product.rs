use crate::backend::{dense, ExecutionContext, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct SubmatrixProduct(pub usize);

impl Operation<ExecutionContext> for SubmatrixProduct {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        let m = self.0;
        if inputs.len() == 2 {
            if inputs[0].cols() == 1 && inputs[1].cols() == 1 {
                if inputs[0].rows() % m == 0 && inputs[1].rows() % m == 0 {
                    let inp1 = Shape::new(m, inputs[0].rows() / m);
                    let inp2 = Shape::new(m, inputs[1].rows() / m);
                    let out = inp1.transpose() * inp2;
                    Ok(Shape::new(out.size(), 1))
                } else {
                    Err(format!(
                        "Input vectors ({}, {}) must have dimension divisible by {m}!",
                        inputs[0].rows(),
                        inputs[1].rows()
                    ))
                }
            } else {
                Err("Input must be a vector!".to_string())
            }
        } else {
            Err(format!("Invalid number of inputs in linear! Expected 2, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        dense::submatrix_product(self.0, inputs[0].values.dense(), inputs[1].values.dense(), output.values.dense_mut());
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        dense::backprop_submatrix_product(
            self.0,
            input1[0].values.dense(),
            input1[0].gradients.as_mut(),
            input2[0].values.dense(),
            input2[0].gradients.as_mut(),
            output.gradients.as_ref().unwrap(),
        );
    }
}
