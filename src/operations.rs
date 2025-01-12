mod activate;
mod affine;
mod affine_dual;
mod concat;
mod conv;
mod gather;
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

use crate::{autograd::Node, tensor::Activation, ConvolutionDescription, GraphBuilder};

pub fn activate(builder: &mut GraphBuilder, input: Node, activation: Activation) -> Node {
    builder.create_result_of_operation(activation, &[input])
}

pub fn add(builder: &mut GraphBuilder, input1: Node, input2: Node) -> Node {
    linear_combination(builder, 1.0, input1, 1.0, input2)
}

pub fn affine(builder: &mut GraphBuilder, weights: Node, input: Node, bias: Node) -> Node {
    builder.create_result_of_operation(affine::Affine(linear::Linear), &[weights, input, bias])
}

pub fn concat(builder: &mut GraphBuilder, input1: Node, input2: Node) -> Node {
    builder.create_result_of_operation(concat::Concat, &[input1, input2])
}

pub fn linear_combination(builder: &mut GraphBuilder, alpha: f32, input1: Node, beta: f32, input2: Node) -> Node {
    builder.create_result_of_operation(linear_comb::LinearCombination(alpha, beta), &[input1, input2])
}

pub fn matmul(builder: &mut GraphBuilder, weights: Node, input: Node) -> Node {
    builder.create_result_of_operation(linear::Linear, &[weights, input])
}

pub fn mpe(builder: &mut GraphBuilder, predicted: Node, target: Node, power: f32) -> Node {
    builder.create_result_of_operation(power_error::AbsPowerError(power), &[predicted, target])
}

pub fn mse(builder: &mut GraphBuilder, predicted: Node, target: Node) -> Node {
    builder.create_result_of_operation(power_error::AbsPowerError(2.0), &[predicted, target])
}

pub fn pairwise_mul(builder: &mut GraphBuilder, input: Node) -> Node {
    builder.create_result_of_operation(pairwise::PairwiseMul(false), &[input])
}

pub fn select(builder: &mut GraphBuilder, input1: Node, input2: Node) -> Node {
    builder.create_result_of_operation(select::Select, &[input1, input2])
}

pub fn sub(builder: &mut GraphBuilder, input1: Node, input2: Node) -> Node {
    linear_combination(builder, 1.0, input1, -1.0, input2)
}

/// Reshapes vectors A, B with shape (n, 1) into (m, n / m) and computes A^T B
pub fn submatrix_product(builder: &mut GraphBuilder, m: usize, input1: Node, input2: Node) -> Node {
    builder.create_result_of_operation(submatrix_product::SubmatrixProduct(m), &[input1, input2])
}

/// This fuses the following operations
///
/// ` stm_accumulator = activate(affine(weights,  stm, bias))`
///
/// `nstm_accumulator = activate(affine(weights, nstm, bias))`
///
/// `out = concat(stm_accumulator, nstm_accumulator)`
pub fn sparse_affine_dual_with_activation(
    builder: &mut GraphBuilder,
    weights: Node,
    stm: Node,
    nstm: Node,
    bias: Node,
    activation: Activation,
) -> Node {
    builder.create_result_of_operation(
        affine_dual::SparseAffineDualWithActivation(activation),
        &[weights, stm, nstm, bias],
    )
}

/// Post sparse-affine-dual, doing pairwise would just elementise-mul the stm and nstm
/// accumulators, which is not what is wanted.
///
/// This will perform the pairwise mul within each accumulator.
pub fn pairwise_mul_post_sparse_affine_dual(builder: &mut GraphBuilder, input: Node) -> Node {
    builder.create_result_of_operation(pairwise::PairwiseMul(true), &[input])
}

pub fn softmax_crossentropy_loss(builder: &mut GraphBuilder, predicted: Node, target: Node) -> Node {
    builder.create_result_of_operation(softmax::SoftmaxCrossEntropyLoss, &[predicted, target])
}

pub fn sparse_softmax_crossentropy_loss_masked(
    builder: &mut GraphBuilder,
    mask: Node,
    predicted: Node,
    target: Node,
) -> Node {
    builder.create_result_of_operation(softmax_sparse::SparseSoftmaxCrossEntropyLoss, &[mask, predicted, target])
}

pub fn slice_rows(builder: &mut GraphBuilder, input: Node, start: usize, end: usize) -> Node {
    builder.create_result_of_operation(slice::SliceRows(start, end), &[input])
}

pub fn convolution(builder: &mut GraphBuilder, filters: Node, input: Node, desc: ConvolutionDescription) -> Node {
    builder.create_result_of_operation(desc, &[filters, input])
}

pub fn mask(builder: &mut GraphBuilder, input: Node, mask: Node) -> Node {
    builder.create_result_of_operation(mask::Mask, &[input, mask])
}

pub fn gather(builder: &mut GraphBuilder, input: Node, indices: Node) -> Node {
    builder.create_result_of_operation(gather::Gather, &[input, indices])
}
