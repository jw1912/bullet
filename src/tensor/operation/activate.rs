use crate::tensor::{DenseMatrix, Shape, Tensor};

/// List of supported activation functions.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Identity = 0,
    ReLU = 1,
    CReLU = 2,
    SCReLU = 3,
    SqrReLU = 4,
    Sigmoid = 5,
}

pub fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 1 {
        Ok(inputs[0])
    } else {
        Err(format!("Invalid number of inputs in activation! Expected 1, got {}", inputs.len()))
    }
}

pub fn forward(activation: Activation, inputs: &[&Tensor], output: &mut Tensor) {
    let input = &inputs[0].values;
    let output = &mut output.values;

    match activation {
        Activation::Identity => panic!("No-op!"),
        Activation::ReLU => DenseMatrix::relu(input.dense(), output.dense_mut()),
        Activation::CReLU => DenseMatrix::crelu(input.dense(), output.dense_mut()),
        Activation::SCReLU => DenseMatrix::screlu(input.dense(), output.dense_mut()),
        Activation::SqrReLU => DenseMatrix::sqrrelu(input.dense(), output.dense_mut()),
        Activation::Sigmoid => DenseMatrix::sigmoid(input.dense(), output.dense_mut()),
    }
}

pub fn backprop(activation: Activation, output: &Tensor, inputs: &mut [&mut Tensor]) {
    let input = &inputs[0].values;
    let input_grad = inputs[0].gradients.as_mut().expect("Must track gradients in activations!");
    let output_grad = output.gradients.as_ref().expect("Must exist!");

    match activation {
        Activation::Identity => panic!("No-op!"),
        Activation::ReLU => DenseMatrix::relu_backward(input.dense(), input_grad, output_grad),
        Activation::CReLU => DenseMatrix::crelu_backward(input.dense(), input_grad, output_grad),
        Activation::SCReLU => DenseMatrix::screlu_backward(input.dense(), input_grad, output_grad),
        Activation::SqrReLU => DenseMatrix::sqrrelu_backward(input.dense(), input_grad, output_grad),
        Activation::Sigmoid => DenseMatrix::sigmoid_backward(input.dense(), input_grad, output_grad),
    }
}
