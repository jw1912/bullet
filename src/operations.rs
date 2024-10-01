mod activate;
mod add;
mod linear;
mod power_error;

pub use activate::Activation;
use diffable::{DiffableOperation, Node};

use crate::{
    backend::ExecutionContext,
    tensor::{Shape, Tensor},
    GraphBuilder,
};

pub fn affine(builder: &mut GraphBuilder, weights: Node, input: Node, bias: Node) -> Node {
    let mul = builder.create_result_of_operation(Operation::Linear, &[weights, input]);
    builder.create_result_of_operation(Operation::Add, &[mul, bias])
}

pub fn activate(builder: &mut GraphBuilder, input: Node, activation: Activation) -> Node {
    builder.create_result_of_operation(Operation::Activate(activation), &[input])
}

pub fn mse(builder: &mut GraphBuilder, predicted: Node, target: Node) -> Node {
    builder.create_result_of_operation(Operation::AbsPowerError(2.0), &[predicted, target])
}

/// All supported operations between tensors in bullet.
#[derive(Clone, Copy, Debug)]
pub enum Operation {
    /// Multiply vector by a matrix
    Linear,
    /// Add two matrices/vectors
    Add,
    /// Activate a matrix/vector
    Activate(Activation),
    /// Calculate `output = sum_i powf(abs(input_a[i] - input_b[i]), power)`
    AbsPowerError(f32),
}

impl DiffableOperation<Tensor, ExecutionContext, Shape> for Operation {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        match self {
            Operation::Linear => linear::output_tensor(inputs),
            Operation::Add => add::output_tensor(inputs),
            Operation::Activate(_) => activate::output_tensor(inputs),
            Operation::AbsPowerError(_) => power_error::output_tensor(inputs),
        }
    }

    fn forward(&self, ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
        match *self {
            Operation::Linear => linear::forward(ctx, inputs, output),
            Operation::Add => add::forward(ctx, inputs, output),
            Operation::Activate(activation) => activate::forward(activation, inputs, output),
            Operation::AbsPowerError(power) => power_error::forward(power, inputs, output),
        }
    }

    fn backward(&self, ctx: &mut ExecutionContext, output_grad: &Tensor, inputs: &mut [&mut Tensor]) {
        match *self {
            Operation::Linear => linear::backprop(ctx, output_grad, inputs),
            Operation::Add => add::backprop(ctx, output_grad, inputs),
            Operation::Activate(activation) => activate::backprop(activation, output_grad, inputs),
            Operation::AbsPowerError(power) => power_error::backprop(power, output_grad, inputs),
        }
    }
}
