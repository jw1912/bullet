mod activate;
mod add;
mod affine;
mod affine_dual;
mod concat;
mod linear;
mod pairwise;
mod power_error;
mod select;
mod softmax;
mod softmax_sparse;

pub use activate::Activation;

use diffable::DiffableOperation;

use crate::{
    backend::ExecutionContext,
    tensor::{Shape, Tensor},
};

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
    /// Concat two vectors
    Concat,
    /// Multiply vector by a matrix
    Linear,
    /// Split vector in two and element-wise multiply the two halves
    PairwiseMul(bool),
    /// Select a subsection of a vector to use
    Select,
    /// Apply softmax followed by crossentropy loss
    SoftmaxCrossEntropyLoss,
    /// Apply sparse masked softmax followed by crossentropy loss
    SparseSoftmaxCrossEntropyLoss,
    /// Warning! Internal use only!
    SparseAffineDual(Activation),
}

impl DiffableOperation<Tensor, ExecutionContext, Shape> for Operation {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        match self {
            Operation::AbsPowerError(_) => power_error::output_tensor(inputs),
            Operation::Activate(_) => activate::output_tensor(inputs),
            Operation::Add => add::output_tensor(inputs),
            Operation::Concat => concat::output_tensor(inputs),
            Operation::Affine => affine::output_tensor(inputs),
            Operation::Linear => linear::output_tensor(inputs),
            Operation::PairwiseMul(_) => pairwise::output_tensor(inputs),
            Operation::Select => select::output_tensor(inputs),
            Operation::SoftmaxCrossEntropyLoss => softmax::output_tensor(inputs),
            Operation::SparseSoftmaxCrossEntropyLoss => softmax_sparse::output_tensor(inputs),
            Operation::SparseAffineDual(_) => affine_dual::output_tensor(inputs),
        }
    }

    fn forward(&self, ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
        match *self {
            Operation::AbsPowerError(power) => power_error::forward(power, inputs, output),
            Operation::Activate(activation) => activate::forward(activation, inputs, output),
            Operation::Add => add::forward(ctx, inputs, output),
            Operation::Affine => affine::forward(ctx, inputs, output),
            Operation::Concat => concat::forward(ctx, inputs, output),
            Operation::Linear => linear::forward(ctx, inputs, output),
            Operation::PairwiseMul(pc) => pairwise::forward(inputs, output, pc),
            Operation::Select => select::forward(inputs, output),
            Operation::SoftmaxCrossEntropyLoss => softmax::forward(ctx, inputs, output),
            Operation::SparseSoftmaxCrossEntropyLoss => softmax_sparse::forward(inputs, output),
            Operation::SparseAffineDual(activation) => affine_dual::forward(inputs, output, activation),
        }
    }

    fn backward(&self, ctx: &mut ExecutionContext, output_grad: &Tensor, inputs: &mut [&mut Tensor]) {
        match *self {
            Operation::AbsPowerError(power) => power_error::backprop(power, output_grad, inputs),
            Operation::Activate(activation) => activate::backprop(activation, output_grad, inputs),
            Operation::Add => add::backprop(ctx, output_grad, inputs),
            Operation::Affine => affine::backprop(ctx, output_grad, inputs),
            Operation::Concat => concat::backprop(ctx, output_grad, inputs),
            Operation::Linear => linear::backprop(ctx, output_grad, inputs),
            Operation::PairwiseMul(pc) => pairwise::backprop(output_grad, inputs, pc),
            Operation::Select => select::backprop(output_grad, inputs),
            Operation::SoftmaxCrossEntropyLoss => softmax::backprop(output_grad, inputs),
            Operation::SparseSoftmaxCrossEntropyLoss => softmax_sparse::backprop(output_grad, inputs),
            Operation::SparseAffineDual(activation) => affine_dual::backprop(output_grad, inputs, activation),
        }
    }
}
