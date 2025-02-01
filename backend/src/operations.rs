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
mod reduce_across_batch;
mod select;
mod slice;
mod softmax;
mod softmax_sparse;
mod submatrix_product;

use std::cell::RefCell;

pub use activate::*;
pub use affine::*;
pub use affine_dual::*;
pub use concat::*;
pub use conv::*;
pub use gather::*;
pub use linear::*;
pub use linear_comb::*;
pub use mask::*;
pub use pairwise::*;
pub use power_error::*;
pub use reduce_across_batch::*;
pub use select::*;
pub use slice::*;
pub use softmax::*;
pub use softmax_sparse::*;
pub use submatrix_product::*;

use crate::backend::{DenseMatrix, Tensor};
use bullet_core::shape::Shape;

fn setup_ones(tensor: &mut Tensor, batch_size: usize) {
    if let Some(ones) = tensor.internal.get_mut("ones") {
        if ones.borrow().shape.size() < batch_size {
            *ones = RefCell::new(DenseMatrix::ones(tensor.values.device(), Shape::new(batch_size, 1)));
        }
    } else {
        let ones = RefCell::new(DenseMatrix::ones(tensor.values.device(), Shape::new(batch_size, 1)));
        tensor.internal.insert("ones".to_string(), ones);
    }
}

fn setup_softmax(tensor: &mut Tensor) {
    if !tensor.internal.contains_key("softmaxed") {
        let zeros = RefCell::new(DenseMatrix::zeroed(tensor.values.device(), Shape::new(1, 1)));
        tensor.internal.insert("softmaxed".to_string(), zeros);
        let zeros = RefCell::new(DenseMatrix::zeroed(tensor.values.device(), Shape::new(1, 1)));
        tensor.internal.insert("individual_losses".to_string(), zeros);
    }
}
