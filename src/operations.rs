mod activate;
mod add;
mod linear;
mod power_error;

pub use activate::Activation;
use diffable::DiffableOperation;

use crate::{
    backend::ExecutionContext,
    tensor::{Shape, Tensor},
};

/// All supported operations between tensors in bullet.
#[derive(Clone, Copy, Debug)]
pub enum Operation {
    /// Multiply vector by a matrix
    Linear,
    /// Add two matrices/vectors
    Add,
    /// Activate a matrix/vector
    Activate(Activation),
    /// Calculate `output[i] = powf(abs(input_a[i] - input_b[i]), power)`
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
