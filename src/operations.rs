use diffable::Node;

use crate::{
    tensor::{Activation, Operation},
    GraphBuilder,
};

pub fn activate(builder: &mut GraphBuilder, input: Node, activation: Activation) -> Node {
    builder.create_result_of_operation(Operation::Activate(activation), &[input])
}

pub fn affine(builder: &mut GraphBuilder, weights: Node, input: Node, bias: Node) -> Node {
    builder.create_result_of_operation(Operation::Affine, &[weights, input, bias])
}

pub fn concat(builder: &mut GraphBuilder, input1: Node, input2: Node) -> Node {
    builder.create_result_of_operation(Operation::Concat, &[input1, input2])
}

pub fn mpe(builder: &mut GraphBuilder, predicted: Node, target: Node, power: f32) -> Node {
    builder.create_result_of_operation(Operation::AbsPowerError(power), &[predicted, target])
}

pub fn mse(builder: &mut GraphBuilder, predicted: Node, target: Node) -> Node {
    builder.create_result_of_operation(Operation::AbsPowerError(2.0), &[predicted, target])
}

pub fn pairwise_mul(builder: &mut GraphBuilder, input: Node) -> Node {
    builder.create_result_of_operation(Operation::PairwiseMul(false), &[input])
}

pub fn select(builder: &mut GraphBuilder, input1: Node, input2: Node) -> Node {
    builder.create_result_of_operation(Operation::Select, &[input1, input2])
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
    builder.create_result_of_operation(Operation::SparseAffineDual(activation), &[weights, stm, nstm, bias])
}

/// Post sparse-affine-dual, doing pairwise would just elementise-mul the stm and nstm
/// accumulators, which is not what is wanted.
///
/// This will perform the pairwise mul within each accumulator.
pub fn pairwise_mul_post_sparse_affine_dual(builder: &mut GraphBuilder, input: Node) -> Node {
    builder.create_result_of_operation(Operation::PairwiseMul(true), &[input])
}

pub fn softmax_crossentropy_loss(builder: &mut GraphBuilder, predicted: Node, target: Node) -> Node {
    builder.create_result_of_operation(Operation::SoftmaxCrossEntropyLoss, &[predicted, target])
}
