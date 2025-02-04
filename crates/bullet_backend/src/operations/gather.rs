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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::backend::Matrix;

    #[test]
    fn test_gather() {
        let device = Arc::new(ExecutionContext::default());

        let shape1 = Shape::new_batched(3, 1, 3);

        let mut inputs = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut output = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut indices = Tensor::new(device.clone(), Shape::new(1, 1), false);

        inputs.load_dense_from_slice(shape1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        unsafe {
            indices.load_sparse_from_slice(Shape::new(5, 1), 5, &[-1, 0, 2, 1, 2]);
        }

        Gather.forward(&[&inputs, &indices], &mut output);

        let buf = output.get_dense_vals().unwrap();
        assert_eq!(&buf, &[0.0, 1.0, 3.0, 2.0, 3.0, 0.0, 4.0, 6.0, 5.0, 6.0, 0.0, 7.0, 9.0, 8.0, 9.0]);

        if let Matrix::Dense(vals) = &output.values {
            vals.copy_into(output.gradients.as_mut().unwrap())
        }

        Gather.backward(&output, &mut [&mut inputs, &mut indices]);

        let mut buf = [0.0; 9];
        inputs.gradients.as_ref().unwrap().write_to_slice(&mut buf);
        assert_eq!(buf, [1.0, 2.0, 6.0, 4.0, 5.0, 12.0, 7.0, 8.0, 18.0,]);
    }
}
