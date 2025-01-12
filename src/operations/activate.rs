use crate::{
    autograd::Operation,
    tensor::{Activation, DenseMatrix, ExecutionContext, Shape, Tensor},
};

impl Operation for Activation {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 1 {
            Ok(inputs[0])
        } else {
            Err(format!("Invalid number of inputs in activation! Expected 1, got {}", inputs.len()))
        }
    }

    fn forward(&self, _: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
        let input = &inputs[0].values;
        let output = &mut output.values;

        match *self {
            Activation::Identity => panic!("No-op!"),
            Activation::ReLU => DenseMatrix::relu(input.dense(), output.dense_mut()),
            Activation::CReLU => DenseMatrix::crelu(input.dense(), output.dense_mut()),
            Activation::SCReLU => DenseMatrix::screlu(input.dense(), output.dense_mut()),
            Activation::SqrReLU => DenseMatrix::sqrrelu(input.dense(), output.dense_mut()),
            Activation::Sigmoid => DenseMatrix::sigmoid(input.dense(), output.dense_mut()),
        }
    }

    fn backward(&self, _: &mut ExecutionContext, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let input = &inputs[0].values;
        let input_grad = inputs[0].gradients.as_mut().expect("Must track gradients in activations!");
        let output_grad = output.gradients.as_ref().expect("Must exist!");

        match *self {
            Activation::Identity => panic!("No-op!"),
            Activation::ReLU => DenseMatrix::relu_backward(input.dense(), input_grad, output_grad),
            Activation::CReLU => DenseMatrix::crelu_backward(input.dense(), input_grad, output_grad),
            Activation::SCReLU => DenseMatrix::screlu_backward(input.dense(), input_grad, output_grad),
            Activation::SqrReLU => DenseMatrix::sqrrelu_backward(input.dense(), input_grad, output_grad),
            Activation::Sigmoid => DenseMatrix::sigmoid_backward(input.dense(), input_grad, output_grad),
        }
    }
}
