use crate::{
    tensor::{DenseMatrix, Shape},
    Tensor,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    ReLU,
    CReLU,
    SCReLU,
    SqrReLU,
    Sigmoid,
}

pub fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 1 {
        Ok(inputs[0])
    } else {
        Err(String::from("Invalid number of inputs!"))
    }
}

pub fn forward(activation: Activation, inputs: &[&Tensor], output: &mut Tensor) {
    let input = &inputs[0].values;
    let output = &mut output.values;

    match activation {
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
        Activation::ReLU => DenseMatrix::relu_backward(input.dense(), input_grad, output_grad),
        Activation::CReLU => DenseMatrix::crelu_backward(input.dense(), input_grad, output_grad),
        Activation::SCReLU => DenseMatrix::screlu_backward(input.dense(), input_grad, output_grad),
        Activation::SqrReLU => DenseMatrix::sqrrelu_backward(input.dense(), input_grad, output_grad),
        Activation::Sigmoid => DenseMatrix::sigmoid_backward(output.values.dense(), input_grad, output_grad),
    }
}