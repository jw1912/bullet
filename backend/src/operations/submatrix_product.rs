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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::backend::Matrix;
    use bullet_core::device::Device;

    #[test]
    fn test_submatrix_product() {
        let device = Arc::new(ExecutionContext::default());

        let shape = Shape::new_batched(2, 1, 3);
        let key_size = 1;

        let mut input1 = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut input2 = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut output = Tensor::new(device.clone(), Shape::new(1, 1), true);

        device.panic_if_device_error("Failed to initialise matrices!");

        // load matrices from CPU
        {
            input1.load_dense_from_slice(shape, &[1.0; 6]);

            input2.load_dense_from_slice(shape, &[2.0, 1.0, 4.0, 3.0, 0.0, 4.0]);

            assert_eq!(input1.shape(), shape);
            assert_eq!(input2.shape(), shape);

            device.panic_if_device_error("Failed to load data from CPU!");
        }

        // normal matmul
        {
            SubmatrixProduct(key_size).forward(&[&input1, &input2], &mut output);

            device.panic_if_device_error("Failed to calculate matmul!");

            assert_eq!(output.shape(), Shape::new_batched(4, 1, 3));

            let buf = output.get_dense_vals().unwrap();
            assert_eq!(&buf, &[2.0, 2.0, 1.0, 1.0, 4.0, 4.0, 3.0, 3.0, 0.0, 0.0, 4.0, 4.0]);

            device.panic_if_device_error("Failed to write data to CPU!");
        }

        if let Matrix::Dense(vals) = &output.values {
            vals.copy_into(output.gradients.as_mut().unwrap());
        }

        // backprop normal matmul
        {
            SubmatrixProduct(key_size).backward(&output, &mut [&mut input1, &mut input2]);

            device.panic_if_device_error("Failed to backprop matmul!");

            assert_eq!(input1.gradients.as_ref().unwrap().shape(), shape);
            assert_eq!(input2.gradients.as_ref().unwrap().shape(), shape);

            let mut grad1 = [0.0; 6];
            input1.gradients.as_ref().unwrap().write_to_slice(&mut grad1);
            assert_eq!(grad1, [5.0, 5.0, 25.0, 25.0, 16.0, 16.0]);

            let mut grad2 = [0.0; 6];
            input2.gradients.as_ref().unwrap().write_to_slice(&mut grad2);
            assert_eq!(grad2, [4.0, 2.0, 8.0, 6.0, 0.0, 8.0]);

            device.panic_if_device_error("Failed to write data to CPU!");
        }
    }
}
