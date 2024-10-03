mod activate;
mod add;
mod affine;
mod linear;
mod pairwise;
mod power_error;
mod select;

pub use activate::Activation;
use diffable::{DiffableOperation, Node};

use crate::{
    backend::ExecutionContext,
    tensor::{Shape, Tensor},
    GraphBuilder,
};

pub fn affine(builder: &mut GraphBuilder, weights: Node, input: Node, bias: Node) -> Node {
    builder.create_result_of_operation(Operation::Affine, &[weights, input, bias])
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
    /// Calculate `output = sum_i powf(abs(input_a[i] - input_b[i]), power)`
    AbsPowerError(f32),
    /// Activate a matrix/vector
    Activate(Activation),
    /// Add two matrices/vectors
    Add,
    /// Apply affine transform
    Affine,
    /// Multiply vector by a matrix
    Linear,
    /// Split vector in two and element-wise multiply the two halves
    PairwiseMul,
    /// Select a subsection of a vector to use
    Select,
}

impl DiffableOperation<Tensor, ExecutionContext, Shape> for Operation {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        match self {
            Operation::AbsPowerError(_) => power_error::output_tensor(inputs),
            Operation::Activate(_) => activate::output_tensor(inputs),
            Operation::Add => add::output_tensor(inputs),
            Operation::Affine => affine::output_tensor(inputs),
            Operation::Linear => linear::output_tensor(inputs),
            Operation::PairwiseMul => pairwise::output_tensor(inputs),
            Operation::Select => select::output_tensor(inputs),
        }
    }

    fn forward(&self, ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
        match *self {
            Operation::AbsPowerError(power) => power_error::forward(power, inputs, output),
            Operation::Activate(activation) => activate::forward(activation, inputs, output),
            Operation::Add => add::forward(ctx, inputs, output),
            Operation::Affine => affine::forward(ctx, inputs, output),
            Operation::Linear => linear::forward(ctx, inputs, output),
            Operation::PairwiseMul => pairwise::forward(inputs, output),
            Operation::Select => select::forward(inputs, output),
        }
    }

    fn backward(&self, ctx: &mut ExecutionContext, output_grad: &Tensor, inputs: &mut [&mut Tensor]) {
        match *self {
            Operation::AbsPowerError(power) => power_error::backprop(power, output_grad, inputs),
            Operation::Activate(activation) => activate::backprop(activation, output_grad, inputs),
            Operation::Add => add::backprop(ctx, output_grad, inputs),
            Operation::Affine => affine::backprop(ctx, output_grad, inputs),
            Operation::Linear => linear::backprop(ctx, output_grad, inputs),
            Operation::PairwiseMul => pairwise::backprop(output_grad, inputs),
            Operation::Select => select::backprop(output_grad, inputs),
        }
    }
}
