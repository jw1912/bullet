mod activate;
mod affine;
mod affine_dual;
mod concat;
mod conv;
mod linear;
mod linear_comb;
mod mask;
mod pairwise;
mod power_error;
mod select;
mod slice;
mod softmax;
mod softmax_sparse;
mod submatrix_product;

pub use activate::Activation;

use crate::{
    backend::{ConvolutionDescription, ExecutionContext},
    tensor::{Shape, Tensor},
};

/// All supported operations between tensors in bullet.
#[derive(Clone, Copy, Debug)]
pub enum Operation {
    /// Calculate `output = sum_i powf(abs(input_a[i] - input_b[i]), power)`
    AbsPowerError(f32),
    /// Activate a matrix/vector
    Activate(Activation),
    /// Apply affine transform
    Affine,
    /// Concat two vectors
    Concat,
    /// Convolution
    Convolution(ConvolutionDescription),
    /// Multiply vector by a matrix
    Linear,
    /// Linear combination of two vectors
    LinearCombination(f32, f32),
    /// Mask a vector, replacing elements not in the mask by 0
    Mask,
    /// Split vector in two and element-wise multiply the two halves
    PairwiseMul(bool),
    /// Select a subsection of a vector to use
    Select,
    /// Take a contiguous slice of rows of a matrix
    SliceRows(usize, usize),
    /// Apply softmax followed by crossentropy loss
    SoftmaxCrossEntropyLoss,
    /// Apply sparse masked softmax followed by crossentropy loss
    SparseSoftmaxCrossEntropyLoss,
    /// Warning! Internal use only!
    SparseAffineDual(Activation),
    /// Reshapes vectors A, B with shape (n, 1) into (m, n / m) and computes A^T B
    SubmatrixProduct(usize),
}

impl Operation {
    pub fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        match *self {
            Operation::AbsPowerError(_) => power_error::output_tensor(inputs),
            Operation::Activate(_) => activate::output_tensor(inputs),
            Operation::Concat => concat::output_tensor(inputs),
            Operation::Convolution(desc) => conv::output_tensor(inputs, &desc),
            Operation::Affine => affine::output_tensor(inputs),
            Operation::Linear => linear::output_tensor(inputs),
            Operation::LinearCombination(_, _) => linear_comb::output_tensor(inputs),
            Operation::Mask => mask::output_tensor(inputs),
            Operation::PairwiseMul(_) => pairwise::output_tensor(inputs),
            Operation::Select => select::output_tensor(inputs),
            Operation::SliceRows(start, end) => slice::output_tensor(inputs, start, end),
            Operation::SoftmaxCrossEntropyLoss => softmax::output_tensor(inputs),
            Operation::SparseSoftmaxCrossEntropyLoss => softmax_sparse::output_tensor(inputs),
            Operation::SparseAffineDual(_) => affine_dual::output_tensor(inputs),
            Operation::SubmatrixProduct(m) => submatrix_product::output_tensor(m, inputs),
        }
    }

    pub fn forward(&self, ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
        match *self {
            Operation::AbsPowerError(power) => power_error::forward(power, inputs, output),
            Operation::Activate(activation) => activate::forward(activation, inputs, output),
            Operation::Affine => affine::forward(ctx, inputs, output),
            Operation::Concat => concat::forward(ctx, inputs, output),
            Operation::Convolution(desc) => conv::forward(ctx, &desc, inputs, output),
            Operation::Linear => linear::forward(ctx, inputs, output),
            Operation::LinearCombination(alpha, beta) => linear_comb::forward(ctx, alpha, beta, inputs, output),
            Operation::Mask => mask::forward(inputs, output),
            Operation::PairwiseMul(pc) => pairwise::forward(inputs, output, pc),
            Operation::Select => select::forward(inputs, output),
            Operation::SliceRows(start, end) => slice::forward(ctx, inputs, start, end, output),
            Operation::SoftmaxCrossEntropyLoss => softmax::forward(ctx, inputs, output),
            Operation::SparseSoftmaxCrossEntropyLoss => softmax_sparse::forward(inputs, output),
            Operation::SparseAffineDual(activation) => affine_dual::forward(inputs, output, activation),
            Operation::SubmatrixProduct(m) => submatrix_product::forward(ctx, m, inputs, output),
        }
    }

    pub fn backward(&self, ctx: &mut ExecutionContext, output_grad: &Tensor, inputs: &mut [&mut Tensor]) {
        match *self {
            Operation::AbsPowerError(power) => power_error::backprop(power, output_grad, inputs),
            Operation::Activate(activation) => activate::backprop(activation, output_grad, inputs),
            Operation::Affine => affine::backprop(ctx, output_grad, inputs),
            Operation::Concat => concat::backprop(ctx, output_grad, inputs),
            Operation::Convolution(desc) => conv::backprop(ctx, &desc, output_grad, inputs),
            Operation::Linear => linear::backprop(ctx, output_grad, inputs),
            Operation::LinearCombination(alpha, beta) => linear_comb::backprop(ctx, alpha, beta, output_grad, inputs),
            Operation::Mask => mask::backprop(output_grad, inputs),
            Operation::PairwiseMul(pc) => pairwise::backprop(output_grad, inputs, pc),
            Operation::Select => select::backprop(output_grad, inputs),
            Operation::SliceRows(start, end) => slice::backprop(ctx, output_grad, start, end, inputs),
            Operation::SoftmaxCrossEntropyLoss => softmax::backprop(output_grad, inputs),
            Operation::SparseSoftmaxCrossEntropyLoss => softmax_sparse::backprop(output_grad, inputs),
            Operation::SparseAffineDual(activation) => affine_dual::backprop(output_grad, inputs, activation),
            Operation::SubmatrixProduct(m) => submatrix_product::backprop(ctx, m, output_grad, inputs),
        }
    }
}
