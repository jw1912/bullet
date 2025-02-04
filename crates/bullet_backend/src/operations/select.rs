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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::backend::Matrix;
    use bullet_core::device::Device;

    #[test]
    fn test_select() {
        let device = Arc::new(ExecutionContext::default());

        let shape1 = Shape::new_batched(8, 1, 3);
        let shape2 = Shape::new_batched(4, 1, 3);

        let mut input1 = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut input2 = Tensor::new(device.clone(), Shape::new(1, 1), false);
        let mut output = Tensor::new(device.clone(), Shape::new(1, 1), true);

        device.panic_if_device_error("Failed to initialise matrices!");

        input1.load_dense_from_slice(
            shape1,
            &[
                -1.0, 4.0, 2.0, -2.0, 0.0, -3.0, 0.0, -3.0, -1.0, 4.0, 2.0, -2.0, 0.0, -3.0, 0.0, -3.0, -1.0, 4.0, 2.0,
                -2.0, 0.0, -3.0, 0.0, -3.0,
            ],
        );

        unsafe {
            input2.load_sparse_from_slice(shape2, 1, &[0, 1, 2]);
        }

        assert_eq!(input1.shape(), shape1);
        assert_eq!(input2.shape(), shape2);

        device.panic_if_device_error("Failed to load data from CPU!");

        Select.forward(&[&input1, &input2], &mut output);

        device.panic_if_device_error("Failed to calculate matmul!");

        assert_eq!(output.shape(), Shape::new_batched(2, 1, 3));

        let buf = output.get_dense_vals().unwrap();
        assert_eq!(&buf, &[-1.0, 4.0, 2.0, -2.0, 0.0, -3.0]);

        device.panic_if_device_error("Failed to write data to CPU!");

        if let Matrix::Dense(vals) = &output.values {
            vals.copy_into(output.gradients.as_mut().unwrap());
        }

        Select.backward(&output, &mut [&mut input1, &mut input2]);

        device.panic_if_device_error("Failed to backprop matmul!");

        assert_eq!(input1.gradients.as_ref().unwrap().shape(), shape1);

        let mut grad1 = [0.0; 24];
        input1.gradients.as_ref().unwrap().write_to_slice(&mut grad1);
        assert_eq!(
            grad1,
            [
                -1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, -3.0, 0.0, 0.0,
            ],
        );

        device.panic_if_device_error("Failed to write data to CPU!");
    }
}
