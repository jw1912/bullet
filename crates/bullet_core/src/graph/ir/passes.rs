pub mod exchange;
pub mod fuse;

use crate::graph::ir::{
    operation::{
        affine::Matmul,
        binary::{LinearCombination, Select},
        unary::{Unary, UnaryOp},
        GraphIROperationCompilable,
    },
    transform::GraphIRTransform,
    BackendMarker, GraphIR, GraphIRError,
};

pub fn search_for_fusion<B: BackendMarker>(
    ir: &GraphIR<B>,
    node: usize,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let data = ir.get(node)?;

    if let Some(op) = &data.parent_operation {
        if let Some(&Select { input, buckets }) = downcast(op) {
            return exchange::add_select(ir, input, buckets, data);
        }

        if let Some(Unary { input, op }) = downcast(op) {
            if let Some(transform) = exchange::unary_concat(ir, *input, *op, data)? {
                return Ok(Some(transform));
            }
        }

        if let Some(Matmul { a, b, transa: false, transb: false }) = downcast(op) {
            return exchange::matmul_concat(ir, *a, *b, data);
        }

        if let Some(LinearCombination { a, b, alpha, beta }) = downcast(op) {
            return fuse::linear_comb(ir, *alpha, a, *beta, b, data);
        }

        if let Some(Unary { input, op: UnaryOp::DiffableFromOutput(act) }) = downcast(op) {
            return fuse::diffable_from_output(ir, input, *act, data);
        }

        if let Some(Unary { input, op: UnaryOp::AbsPow(x) }) = downcast(op) {
            return fuse::power_error(ir, input, *x, data);
        }

        if let Some(Unary { input, op: UnaryOp::Mul(x) }) = downcast(op) {
            return fuse::scale(ir, input, *x, data);
        }
    }

    Ok(None)
}

#[allow(clippy::borrowed_box)]
pub fn downcast<B: BackendMarker, T: 'static>(op: &Box<dyn GraphIROperationCompilable<B>>) -> Option<&T> {
    let op: &dyn std::any::Any = op.as_ref();
    op.downcast_ref()
}
