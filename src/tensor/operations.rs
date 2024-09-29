mod add;
mod linear;

use diffable::{DiffableOperation, GraphBuilder, Node};

use crate::backend::ExecutionContext;

use super::{Shape, Tensor};

/// All supported operations between tensors in bullet.
#[derive(Clone, Copy, Debug)]
pub enum Operation {
    /// Multiply vector by a matrix
    Linear,
    /// Add two matrices/vectors
    Add,
}

impl Operation {
    pub fn linear(builder: &mut GraphBuilder<Tensor>, a: Node, b: Node) -> Node {
        builder.create_result_of_operation(Operation::Linear, &[a, b])
    }

    pub fn add(builder: &mut GraphBuilder<Tensor>, a: Node, b: Node) -> Node {
        builder.create_result_of_operation(Operation::Add, &[a, b])
    }
}

impl DiffableOperation<Tensor, ExecutionContext, Shape> for Operation {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        match self {
            Operation::Linear => linear::output_tensor(inputs),
            Operation::Add => add::output_tensor(inputs),
        }
    }

    fn forward(&self, ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
        match self {
            Operation::Linear => linear::forward(ctx, inputs, output),
            Operation::Add => add::forward(ctx, inputs, output),
        }
    }

    fn backward(&self, ctx: &mut ExecutionContext, output_grad: &Tensor, inputs: &mut [&mut Tensor]) {
        match self {
            Operation::Linear => linear::backprop(ctx, output_grad, inputs),
            Operation::Add => add::backprop(ctx, output_grad, inputs),
        }
    }
}
