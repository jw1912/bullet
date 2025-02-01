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
