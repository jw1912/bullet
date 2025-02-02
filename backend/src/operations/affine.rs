use crate::backend::{dense, sparse, ExecutionContext, Matrix, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

use super::linear::Linear;

#[derive(Debug)]
pub struct Affine(pub Linear);

impl Operation<ExecutionContext> for Affine {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 3 && inputs[0] * inputs[1] == inputs[2] {
            Ok(inputs[2])
        } else {
            Err(format!("Invalid number of inputs in affine! Expected 3, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        let weights = inputs[0].values.dense();
        let biases = inputs[2].values.dense();

        let batch_size = inputs[1].values.shape().batch_size().unwrap_or(1);
        super::setup_ones(output, batch_size);
        let ones = output.internal.get("ones").unwrap().borrow();
        let out = output.values.dense_mut();

        match &inputs[1].values {
            Matrix::Sparse(sparse) => sparse::affine(weights, sparse, Some(biases), out),
            Matrix::Dense(dense) => {
                dense::matmul(weights, false, dense, false, out);
                dense::add_assign_single_to_batched_scaled(&ones.buf, 1.0, biases, out);
            }
        }
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, inputs2) = inputs.split_at_mut(1);
        let (input2, input3) = inputs2.split_at_mut(1);
        let out = output.gradients.as_ref().unwrap();

        let ones = output.internal.get("ones").unwrap().borrow();
        assert!(ones.shape.size() >= output.shape().cols());

        match &input2[0].values {
            Matrix::Sparse(sparse) => {
                let input3_values = if input3[0].gradients.is_some() { Some(input3[0].values.dense()) } else { None };

                sparse::backprop_affine(
                    input1[0].values.dense(),
                    input1[0].gradients.as_mut().unwrap(),
                    sparse,
                    input3_values,
                    input3[0].gradients.as_mut(),
                    output.values.dense(),
                    out,
                );
            }
            Matrix::Dense(_) => {
                self.0.backward(output, &mut [&mut input1[0], &mut input2[0]]);
                if let Some(grad) = &mut input3[0].gradients {
                    dense::backprop_add_single(&ones.buf, 1.0, input3[0].values.dense(), grad, out);
                }
            }
        }
    }
}
