use crate::graph::ir::{
    node::AnnotatedNode,
    operation::{
        affine::{Affine, Matmul},
        binary::{AbsPowerError, Concat, FusedPairwiseMulConcat},
        nary::LinearCombination,
        sparse::SparseAffineActivate,
        unary::{DiffableFromOutput, PairwiseMul, Unary, UnaryOp},
    },
    passes::GraphIRSimplePass,
    transform::GraphIRTransform,
    BackendMarker, GraphIR, GraphIRError, GraphIRNode,
};

use super::downcast;

pub struct FusePairwiseMulWithConcat;

impl<B: BackendMarker> GraphIRSimplePass<B> for FusePairwiseMulWithConcat {
    fn try_pass_on_node(&self, ir: &GraphIR<B>, node: usize) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
        let old_data = ir.get(node)?;

        if let Some(Concat { a, b }) = downcast(&old_data.parent_operation) {
            let a_data = ir.get(a.idx)?;
            let b_data = ir.get(b.idx)?;

            if a_data.num_children == 1 && b_data.num_children == 1 {
                if let (Some(&PairwiseMul { input: x }), Some(&PairwiseMul { input: y })) =
                    (downcast(&a_data.parent_operation), downcast(&b_data.parent_operation))
                {
                    if x.idx != y.idx {
                        let new_data = old_data.with_new_op(FusedPairwiseMulConcat { a: x, b: y });
                        return GraphIRTransform::new([a.idx, b.idx], [new_data]);
                    }
                }
            }
        }

        Ok(None)
    }
}

pub struct FuseSparseMatmulWithAdd;

impl<B: BackendMarker> GraphIRSimplePass<B> for FuseSparseMatmulWithAdd {
    fn try_pass_on_node(&self, ir: &GraphIR<B>, node: usize) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
        let old_data = ir.get(node)?;

        if let Some(LinearCombination { items, shape }) = downcast(&old_data.parent_operation) {
            if let &[(a, 1.0), (b, 1.0)] = &items[..] {
                let a = AnnotatedNode { idx: a, shape: *shape };
                let b = AnnotatedNode { idx: b, shape: *shape };

                if let Some(fusion_data) = add_single_sparse(ir, a, b, old_data)? {
                    return Ok(Some(fusion_data));
                }

                if let Some(fusion_data) = add_single_sparse(ir, b, a, old_data)? {
                    return Ok(Some(fusion_data));
                }
            }
        }

        Ok(None)
    }
}

fn add_single_sparse<B: BackendMarker>(
    ir: &GraphIR<B>,
    lhs: AnnotatedNode,
    rhs: AnnotatedNode,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(lhs.idx)?;

    if ir_node.num_children == 1 {
        if let Some(&SparseAffineActivate {
            weights,
            indices,
            values,
            biases: None,
            activation: DiffableFromOutput::Identity,
        }) = downcast(&ir_node.parent_operation)
        {
            let new_data = old_data.with_new_op(SparseAffineActivate {
                weights,
                indices,
                values,
                biases: Some(rhs),
                activation: DiffableFromOutput::Identity,
            });
            return GraphIRTransform::new([lhs.idx], [new_data]);
        }
    }

    Ok(None)
}

pub struct FuseSparseAffineWithDiffableFromOutput;

impl<B: BackendMarker> GraphIRSimplePass<B> for FuseSparseAffineWithDiffableFromOutput {
    fn try_pass_on_node(&self, ir: &GraphIR<B>, node: usize) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
        let old_data = ir.get(node)?;

        if let Some(&Unary { input, op: UnaryOp::DiffableFromOutput(activation) }) =
            downcast(&old_data.parent_operation)
        {
            let ir_node = ir.get(input.idx)?;

            if ir_node.num_children == 1 {
                if let Some(&SparseAffineActivate {
                    weights,
                    biases,
                    values,
                    indices,
                    activation: DiffableFromOutput::Identity,
                }) = downcast(&ir_node.parent_operation)
                {
                    let new_data =
                        old_data.with_new_op(SparseAffineActivate { weights, biases, values, indices, activation });
                    return GraphIRTransform::new([input.idx], vec![new_data]);
                }
            }
        }

        Ok(None)
    }
}

pub struct LowPriorityFusions;

impl<B: BackendMarker> GraphIRSimplePass<B> for LowPriorityFusions {
    fn try_pass_on_node(&self, ir: &GraphIR<B>, node: usize) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
        let data = ir.get(node)?;

        if let Some(LinearCombination { items, shape }) = downcast(&data.parent_operation) {
            if let [(a, alpha), (b, beta)] = &items[..] {
                let a = AnnotatedNode { idx: *a, shape: *shape };
                let b = AnnotatedNode { idx: *b, shape: *shape };

                if *alpha == 1.0 && *beta == 1.0 {
                    if let Some(fusion_data) = add_single_dense(ir, a, b, data)? {
                        return Ok(Some(fusion_data));
                    }

                    if let Some(fusion_data) = add_single_dense(ir, b, a, data)? {
                        return Ok(Some(fusion_data));
                    }
                }

                if let Some(fusion_data) = linear_comb_single(ir, *alpha, a, *beta, b, data)? {
                    return Ok(Some(fusion_data));
                }

                if let Some(fusion_data) = linear_comb_single(ir, *beta, b, *alpha, a, data)? {
                    return Ok(Some(fusion_data));
                }
            }

            return compact_linear_comb(ir, data);
        }

        if let Some(Unary { input, op: UnaryOp::AbsPow(x) }) = downcast(&data.parent_operation) {
            return power_error(ir, *input, *x, data);
        }

        if let Some(Unary { input, op: UnaryOp::Mul(x) }) = downcast(&data.parent_operation) {
            return scale(ir, *input, *x, data);
        }

        Ok(None)
    }
}

fn power_error<B: BackendMarker>(
    ir: &GraphIR<B>,
    node: AnnotatedNode,
    power: f32,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.num_children == 1 {
        if let Some(LinearCombination { items, shape }) = downcast(&ir_node.parent_operation) {
            if let [(a, 1.0), (b, -1.0)] = items[..] {
                let a = AnnotatedNode { idx: a, shape: *shape };
                let b = AnnotatedNode { idx: b, shape: *shape };

                if a.idx != b.idx && ir.get(a.idx)?.info.batched == ir.get(b.idx)?.info.batched {
                    let new_data = old_data.with_new_op(AbsPowerError { a, b, power });
                    return GraphIRTransform::new([node.idx], [new_data]);
                }
            }
        }
    }

    Ok(None)
}

fn scale<B: BackendMarker>(
    ir: &GraphIR<B>,
    node: AnnotatedNode,
    scale: f32,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.num_children == 1 {
        if let Some(LinearCombination { items, shape }) = downcast(&ir_node.parent_operation) {
            if let [(a, alpha), (b, beta)] = items[..] {
                let a = AnnotatedNode { idx: a, shape: *shape };
                let b = AnnotatedNode { idx: b, shape: *shape };

                let new_data = old_data.with_new_op(LinearCombination::new([(a, alpha * scale), (b, beta * scale)])?);
                return GraphIRTransform::new([node.idx], [new_data]);
            }
        }
    }

    Ok(None)
}

fn add_single_dense<B: BackendMarker>(
    ir: &GraphIR<B>,
    lhs: AnnotatedNode,
    rhs: AnnotatedNode,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(lhs.idx)?;

    if ir_node.num_children == 1 {
        if let Some(&Matmul { a, transa: false, b, transb: false }) = downcast(&ir_node.parent_operation) {
            if !ir.get(rhs.idx)?.info.batched {
                let new_data = old_data.with_new_op(Affine { weights: a, inputs: b, biases: rhs });
                return GraphIRTransform::new([lhs.idx], [new_data]);
            }
        }
    }

    Ok(None)
}

fn linear_comb_single<B: BackendMarker>(
    ir: &GraphIR<B>,
    alpha: f32,
    lhs: AnnotatedNode,
    beta: f32,
    rhs: AnnotatedNode,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(lhs.idx)?;

    if ir_node.num_children == 1 {
        if let Some(&Unary { input, op: UnaryOp::Mul(x) }) = downcast(&ir_node.parent_operation) {
            let new_data = old_data.with_new_op(LinearCombination::new([(input, alpha * x), (rhs, beta)])?);
            return GraphIRTransform::new([lhs.idx], vec![new_data]);
        }
    }

    Ok(None)
}

fn compact_linear_comb<B: BackendMarker>(
    ir: &GraphIR<B>,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let LinearCombination { items, shape } = downcast(&old_data.parent_operation).unwrap();
    let shape = *shape;

    for &(node, weight) in items {
        let ir_node = ir.get(node)?;

        if let Some(LinearCombination { items: par_items, shape: shape2 }) = downcast(&ir_node.parent_operation) {
            assert_eq!(shape.size(), shape2.size());

            let mut items: Vec<_> = items
                .iter()
                .copied()
                .filter_map(|(idx, weight)| (idx != node).then_some((AnnotatedNode { idx, shape }, weight)))
                .collect();

            for &(par_node, par_weight) in par_items {
                items.push((AnnotatedNode { idx: par_node, shape }, par_weight * weight));
            }

            let new_data = old_data.with_new_op(LinearCombination::new(items)?);
            return GraphIRTransform::new([node], [new_data]);
        }
    }

    Ok(None)
}
