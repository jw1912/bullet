use crate::backend::{dense, Activation, ExecutionContext, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct Activate(pub Activation);

impl Operation<ExecutionContext> for Activate {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 1 {
            Ok(inputs[0])
        } else {
            Err(format!("Invalid number of inputs in activation! Expected 1, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        let input = &inputs[0].values;
        let output = &mut output.values;

        match self.0 {
            Activation::Identity => panic!("No-op!"),
            Activation::ReLU => dense::relu(input.dense(), output.dense_mut()),
            Activation::CReLU => dense::crelu(input.dense(), output.dense_mut()),
            Activation::SCReLU => dense::screlu(input.dense(), output.dense_mut()),
            Activation::SqrReLU => dense::sqrrelu(input.dense(), output.dense_mut()),
            Activation::Sigmoid => dense::sigmoid(input.dense(), output.dense_mut()),
        }
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let input = &inputs[0].values;
        let input_grad = inputs[0].gradients.as_mut().expect("Must track gradients in activations!");
        let output_grad = output.gradients.as_ref().expect("Must exist!");

        match self.0 {
            Activation::Identity => panic!("No-op!"),
            Activation::ReLU => dense::relu_backward(input.dense(), input_grad, output_grad),
            Activation::CReLU => dense::crelu_backward(input.dense(), input_grad, output_grad),
            Activation::SCReLU => dense::screlu_backward(input.dense(), input_grad, output_grad),
            Activation::SqrReLU => dense::sqrrelu_backward(input.dense(), input_grad, output_grad),
            Activation::Sigmoid => dense::sigmoid_backward(input.dense(), input_grad, output_grad),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::backend::Matrix;
    use bullet_core::device::Device;

    fn activation_test(act: Activation, fwd_expected: [f32; 4], bwd_expected: [f32; 4]) {
        let device = Arc::new(ExecutionContext::default());

        let shape = Shape::new(2, 2);

        let mut input = Tensor::new(device.clone(), shape, true);
        let mut output = Tensor::new(device.clone(), shape, true);

        device.panic_if_device_error("Failed to initialise matrices!");
        input.load_dense_from_slice(shape, &[-1.0, 0.5, 2.0, -2.0]);
        assert_eq!(input.shape(), shape);

        device.panic_if_device_error("Failed to load data from CPU!");

        Activate(act).forward(&[&input], &mut output);

        if let Matrix::Dense(vals) = &output.values {
            vals.copy_into(output.gradients.as_mut().unwrap());
        }

        device.panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape(), input.shape());

        let buf = output.get_dense_vals().unwrap();
        assert_eq!(&buf, &fwd_expected);

        device.panic_if_device_error("Failed to write data to CPU!");

        Activate(act).backward(&output, &mut [&mut input]);

        device.panic_if_device_error("Failed to backprop activation!");

        let mut buf = [0.0; 4];
        input.gradients.unwrap().write_to_slice(&mut buf);
        assert_eq!(buf, bwd_expected);

        device.panic_if_device_error("Failed to write data to CPU!");
    }

    #[test]
    fn relu() {
        activation_test(Activation::ReLU, [0.0, 0.5, 2.0, 0.0], [0.0, 0.5, 2.0, 0.0]);
    }

    #[test]
    fn crelu() {
        activation_test(Activation::CReLU, [0.0, 0.5, 1.0, 0.0], [0.0, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn screlu() {
        activation_test(Activation::SCReLU, [0.0, 0.25, 1.0, 0.0], [0.0, 0.25, 0.0, 0.0]);
    }

    #[test]
    fn sqrrelu() {
        activation_test(Activation::SqrReLU, [0.0, 0.25, 4.0, 0.0], [0.0, 0.25, 16.0, 0.0]);
    }
}
