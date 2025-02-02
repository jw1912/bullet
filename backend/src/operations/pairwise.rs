use crate::backend::{dense, ExecutionContext, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct PairwiseMul(pub bool);

impl Operation<ExecutionContext> for PairwiseMul {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 1 {
            let input = inputs[0];
            if input.rows() % 2 != 0 {
                Err(String::from("Input size must be even!"))
            } else {
                Ok(Shape::new(input.rows() / 2, input.cols()))
            }
        } else {
            Err(format!("Invalid number of inputs in pairwise! Expected 1, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        dense::pairwise(inputs[0].values.dense(), output.values.dense_mut(), self.0);
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let input = inputs[0].values.dense();
        let output_grad = output.gradients.as_ref().expect("Must exist!");
        if let Some(input_grad) = inputs[0].gradients.as_mut() {
            dense::backprop_pairwise(input, output_grad, input_grad, self.0);
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
    fn pairwise_no_concat() {
        let device = Arc::new(ExecutionContext::default());

        let shape1 = Shape::new_batched(4, 1, 2);
        let shape2 = Shape::new_batched(2, 1, 2);

        let mut input = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut output = Tensor::new(device.clone(), Shape::new(1, 1), true);

        device.panic_if_device_error("Failed to initialise matrices!");

        // load matrix from CPU
        input.load_dense_from_slice(shape1, &[-1.0, 0.5, 2.0, 2.0, 2.0, -2.0, 2.0, 2.0]);
        assert_eq!(input.shape(), shape1);

        device.panic_if_device_error("Failed to load data from CPU!");

        PairwiseMul(false).forward(&[&input], &mut output);

        device.panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape(), shape2);

        let buf = output.get_dense_vals().unwrap();
        assert_eq!(&buf, &[-2.0, 1.0, 4.0, -4.0]);

        device.panic_if_device_error("Failed to write data to CPU!");

        if let Matrix::Dense(vals) = &output.values {
            vals.copy_into(output.gradients.as_mut().unwrap());
        }

        PairwiseMul(false).backward(&output, &mut [&mut input]);

        device.panic_if_device_error("Failed to backprop activation!");

        assert_eq!(input.gradients.as_ref().unwrap().shape(), input.shape());

        let mut buf = [0.0; 8];
        input.gradients.as_ref().unwrap().write_to_slice(&mut buf);
        assert_eq!(buf, [-4.0, 2.0, 2.0, 0.5, 8.0, -8.0, 8.0, 8.0]);

        device.panic_if_device_error("Failed to write data to CPU!");
    }

    #[test]
    fn pairwise_post_concat() {
        let device = Arc::new(ExecutionContext::default());

        let shape1 = Shape::new_batched(4, 1, 2);
        let shape2 = Shape::new_batched(2, 1, 2);

        let mut input = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut output = Tensor::new(device.clone(), Shape::new(1, 1), true);

        device.panic_if_device_error("Failed to initialise matrices!");

        // load matrix from CPU
        input.load_dense_from_slice(shape1, &[-1.0, 0.5, 2.0, 2.0, 2.0, -2.0, 2.0, 2.0]);
        assert_eq!(input.shape(), shape1);

        device.panic_if_device_error("Failed to load data from CPU!");

        PairwiseMul(true).forward(&[&input], &mut output);

        device.panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape(), shape2);

        let buf = output.get_dense_vals().unwrap();
        assert_eq!(&buf, &[-0.5, 4.0, -4.0, 4.0]);

        device.panic_if_device_error("Failed to write data to CPU!");

        if let Matrix::Dense(vals) = &output.values {
            vals.copy_into(output.gradients.as_mut().unwrap());
        }

        PairwiseMul(true).backward(&output, &mut [&mut input]);

        device.panic_if_device_error("Failed to backprop activation!");

        assert_eq!(input.gradients.as_ref().unwrap().shape(), input.shape());

        let mut buf = [0.0; 8];
        input.gradients.as_ref().unwrap().write_to_slice(&mut buf);
        assert_eq!(buf, [-0.25, 0.5, 8.0, 8.0, 8.0, -8.0, 8.0, 8.0]);

        device.panic_if_device_error("Failed to write data to CPU!");
    }
}
