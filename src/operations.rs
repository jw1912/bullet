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

pub use affine::*;
pub use affine_dual::*;
pub use concat::*;
pub use gather::*;
pub use linear::*;
pub use linear_comb::*;
pub use mask::*;
pub use pairwise::*;
pub use power_error::*;
pub use select::*;
pub use slice::*;
pub use softmax::*;
pub use softmax_sparse::*;
pub use submatrix_product::*;

use crate::{autograd::{GraphBuilder, Node}, tensor::Activation, ConvolutionDescription};

/// Reshapes vectors A, B with shape (n, 1) into (m, n / m) and computes A^T B
pub fn submatrix_product(builder: &mut GraphBuilder, m: usize, input1: Node, input2: Node) -> Node {
    builder.create_result_of_operation(SubmatrixProduct(m), &[input1, input2])
}

pub fn softmax_crossentropy_loss(builder: &mut GraphBuilder, predicted: Node, target: Node) -> Node {
    builder.create_result_of_operation(SoftmaxCrossEntropyLoss, &[predicted, target])
}

pub fn sparse_softmax_crossentropy_loss_masked(
    builder: &mut GraphBuilder,
    mask: Node,
    predicted: Node,
    target: Node,
) -> Node {
    builder.create_result_of_operation(SparseSoftmaxCrossEntropyLoss, &[mask, predicted, target])
}

pub fn slice_rows(builder: &mut GraphBuilder, input: Node, start: usize, end: usize) -> Node {
    builder.create_result_of_operation(SliceRows(start, end), &[input])
}

pub fn convolution(builder: &mut GraphBuilder, filters: Node, input: Node, desc: ConvolutionDescription) -> Node {
    builder.create_result_of_operation(desc, &[filters, input])
}

pub fn mask(builder: &mut GraphBuilder, input: Node, mask: Node) -> Node {
    builder.create_result_of_operation(Mask, &[input, mask])
}

pub fn gather(builder: &mut GraphBuilder, input: Node, indices: Node) -> Node {
    builder.create_result_of_operation(Gather, &[input, indices])
}
