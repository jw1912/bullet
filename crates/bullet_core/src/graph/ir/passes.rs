pub mod exchange;
pub mod fuse;

use crate::graph::ir::{
    node::AnnotatedNode,
    operation::{
        affine::Matmul,
        binary::Select,
        nary::LinearCombination,
        unary::{Unary, UnaryOp},
        GraphIROperationCompilable,
    },
    transform::GraphIRTransform,
    BackendMarker, GraphIR, GraphIRError,
};

pub trait SimplePass<B: BackendMarker> {
    fn try_pass(ir: &GraphIR<B>, node: usize) -> Result<Option<GraphIRTransform<B>>, GraphIRError>;
}

pub struct HighPriority;

impl<B: BackendMarker> SimplePass<B> for HighPriority {
    fn try_pass(ir: &GraphIR<B>, node: usize) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
        let data = ir.get(node)?;

        if let Some(op) = &data.parent_operation {
            if let Some(&Select { input, buckets }) = downcast(op) {
                return exchange::select(ir, input, buckets, data);
            }

            if let Some(Unary { input, op }) = downcast(op) {
                if let Some(transform) = exchange::unary_concat(ir, *input, *op, data)? {
                    return Ok(Some(transform));
                }
            }

            if let Some(Matmul { a, b, transa: false, transb: false }) = downcast(op) {
                return exchange::matmul_concat(ir, *a, *b, data);
            }

            if let Some(LinearCombination { items, shape }) = downcast(op) {
                if let [(a, alpha), (b, beta)] = &items[..] {
                    let a = AnnotatedNode { idx: *a, shape: *shape };
                    let b = AnnotatedNode { idx: *b, shape: *shape };

                    if *alpha == 1.0 && *beta == 1.0 {
                        if let Some(fusion_data) = fuse::add_single_sparse(ir, a, b, data)? {
                            return Ok(Some(fusion_data));
                        }

                        if let Some(fusion_data) = fuse::add_single_sparse(ir, b, a, data)? {
                            return Ok(Some(fusion_data));
                        }
                    }
                }
            }

            if let Some(Unary { input, op: UnaryOp::DiffableFromOutput(act) }) = downcast(op) {
                return fuse::diffable_from_output(ir, *input, *act, data);
            }
        }

        Ok(None)
    }
}

pub struct LowPriority;

impl<B: BackendMarker> SimplePass<B> for LowPriority {
    fn try_pass(ir: &GraphIR<B>, node: usize) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
        let data = ir.get(node)?;

        if let Some(op) = &data.parent_operation {
            if let Some(LinearCombination { items, shape }) = downcast(op) {
                if let [(a, alpha), (b, beta)] = &items[..] {
                    let a = AnnotatedNode { idx: *a, shape: *shape };
                    let b = AnnotatedNode { idx: *b, shape: *shape };

                    if *alpha == 1.0 && *beta == 1.0 {
                        if let Some(fusion_data) = fuse::add_single_dense(ir, a, b, data)? {
                            return Ok(Some(fusion_data));
                        }

                        if let Some(fusion_data) = fuse::add_single_dense(ir, b, a, data)? {
                            return Ok(Some(fusion_data));
                        }
                    }

                    if let Some(fusion_data) = fuse::linear_comb_single(ir, *alpha, a, *beta, b, data)? {
                        return Ok(Some(fusion_data));
                    }

                    if let Some(fusion_data) = fuse::linear_comb_single(ir, *beta, b, *alpha, a, data)? {
                        return Ok(Some(fusion_data));
                    }
                }

                return fuse::compact_linear_comb(ir, data);
            }

            if let Some(Unary { input, op: UnaryOp::AbsPow(x) }) = downcast(op) {
                return fuse::power_error(ir, *input, *x, data);
            }

            if let Some(Unary { input, op: UnaryOp::Mul(x) }) = downcast(op) {
                return fuse::scale(ir, *input, *x, data);
            }
        }

        Ok(None)
    }
}

#[allow(clippy::borrowed_box)]
pub fn downcast<B: BackendMarker, T: 'static>(op: &Box<dyn GraphIROperationCompilable<B>>) -> Option<&T> {
    let op: &dyn std::any::Any = op.as_ref();
    op.downcast_ref()
}
