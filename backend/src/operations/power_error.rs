use crate::backend::{dense, ExecutionContext, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct AbsPowerError(pub f32);

impl Operation<ExecutionContext> for AbsPowerError {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 2 && inputs[0] == inputs[1] {
            Ok(Shape::new(1, 1))
        } else {
            Err(format!("Invalid number of inputs in power error! Expected 1, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        dense::abs_power_error(self.0, inputs[0].values.dense(), inputs[1].values.dense(), output.values.dense_mut());
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        dense::backprop_abs_power_error(
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
    use bullet_core::device::Device;

    #[test]
    fn abs_power_error_basic() {
        abs_power_error_custom([-1.0, 4.0, 2.0], [1.0, 2.0, 3.0], [2.0, 2.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]);
    }

    #[test]
    fn abs_power_error_rev() {
        abs_power_error_custom([1.0, 2.0, 3.0], [-1.0, 4.0, 2.0], [2.0, 2.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]);
    }

    fn abs_power_error_custom(
        input_a: [f32; 3],
        input_b: [f32; 3],
        outputs: [f32; 3],
        grad_a: [f32; 3],
        grad_b: [f32; 3],
    ) {
        let device = Arc::new(ExecutionContext::default());
        let shape = Shape::new(3, 1);

        let mut input1 = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut input2 = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut output = Tensor::new(device.clone(), Shape::new(1, 1), true);

        device.panic_if_device_error("Failed to initialise matrices!");

        // load matrices from CPU
        {
            input1.load_dense_from_slice(shape, &input_a);

            input2.load_dense_from_slice(shape, &input_b);

            assert_eq!(input1.shape(), shape);
            assert_eq!(input2.shape(), shape);

            device.panic_if_device_error("Failed to load data from CPU!");
        }

        // power error
        {
            AbsPowerError(1.0).forward(&[&input1, &input2], &mut output);

            device.panic_if_device_error("Failed to add matrices!");

            assert_eq!(output.shape(), shape);

            let buf = output.get_dense_vals().unwrap();
            assert_eq!(&buf, &outputs);

            device.panic_if_device_error("Failed to write data to CPU!");
        }

        // backprop add
        {
            output.gradients.as_mut().unwrap().load_from_slice(shape, &[1.0, 1.0, 1.0]);

            device.panic_if_device_error("Failed to load data from CPU!");

            AbsPowerError(1.0).backward(&output, &mut [&mut input1, &mut input2]);

            device.panic_if_device_error("Failed to backprop addition!");

            assert_eq!(input1.gradients.as_ref().unwrap().shape(), shape);
            assert_eq!(input2.gradients.as_ref().unwrap().shape(), shape);

            let mut grad1 = [0.0; 3];
            input1.gradients.as_ref().unwrap().write_to_slice(&mut grad1);
            assert_eq!(grad1, grad_a);

            let mut grad2 = [0.0; 3];
            input2.gradients.as_ref().unwrap().write_to_slice(&mut grad2);
            assert_eq!(grad2, grad_b);

            device.panic_if_device_error("Failed to write data to CPU!");
        }
    }
}
