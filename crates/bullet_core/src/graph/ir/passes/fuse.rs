use acyclib::graph::NodeId;

use crate::graph::ir::{
    BackendMarker, GraphIR, GraphIRError, GraphIRMethods,
    node::AnnotatedNode,
    operation::{
        GraphIROperationCompilable,
        affine::{Affine, Matmul},
        binary::{AbsPowerError, Concat, FusedPairwiseMulConcat},
        nary::LinearCombination,
        sparse::SparseAffineActivate,
        unary::{DiffableFromOutput, PairwiseMul, Unary, UnaryOp},
    },
    passes::GraphIRSimplePass,
};

use super::downcast;

#[derive(Debug)]
pub struct FusePairwiseMulWithConcat;

impl<B: BackendMarker> GraphIRSimplePass<B> for FusePairwiseMulWithConcat {
    fn try_pass_on_node(&self, ir: &mut GraphIR<B>, target: NodeId) -> Result<bool, GraphIRError> {
        let op = ir.get(target)?.op();

        if let Some(Concat { a, b }) = downcast(op) {
            let a_data = ir.get(a.idx)?;
            let b_data = ir.get(b.idx)?;

            if a_data.children() == 1 && b_data.children() == 1 {
                if let (Some(PairwiseMul { input: x }), Some(PairwiseMul { input: y })) =
                    (downcast(a_data.op()), downcast(b_data.op()))
                {
                    if x.idx != y.idx {
                        ir.replace(target, FusedPairwiseMulConcat { a: x, b: y })?;
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }
}

#[derive(Debug)]
pub struct FuseSparseMatmulWithAdd;

impl<B: BackendMarker> GraphIRSimplePass<B> for FuseSparseMatmulWithAdd
where
    SparseAffineActivate: GraphIROperationCompilable<B>,
{
    fn try_pass_on_node(&self, ir: &mut GraphIR<B>, target: NodeId) -> Result<bool, GraphIRError> {
        let op = ir.get(target)?.op();

        if let Some(LinearCombination { items, shape }) = downcast(op) {
            if let &[(a, 1.0), (b, 1.0)] = &items[..] {
                let a = AnnotatedNode { idx: a, shape };
                let b = AnnotatedNode { idx: b, shape };

                if add_single_sparse(ir, a, b, target)? {
                    return Ok(true);
                }

                if add_single_sparse(ir, b, a, target)? {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

fn add_single_sparse<B: BackendMarker>(
    ir: &mut GraphIR<B>,
    lhs: AnnotatedNode,
    rhs: AnnotatedNode,
    target: NodeId,
) -> Result<bool, GraphIRError>
where
    SparseAffineActivate: GraphIROperationCompilable<B>,
{
    let ir_node = ir.get(lhs.idx)?;

    if ir_node.children() == 1 {
        if let Some(SparseAffineActivate {
            weights,
            indices,
            values,
            biases: None,
            activation: DiffableFromOutput::Identity,
        }) = downcast(ir_node.op())
        {
            ir.replace(
                target,
                SparseAffineActivate {
                    weights,
                    indices,
                    values,
                    biases: Some(rhs),
                    activation: DiffableFromOutput::Identity,
                },
            )?;
            return Ok(true);
        }
    }

    Ok(false)
}

#[derive(Debug)]
pub struct FuseSparseAffineWithDiffableFromOutput;

impl<B: BackendMarker> GraphIRSimplePass<B> for FuseSparseAffineWithDiffableFromOutput
where
    SparseAffineActivate: GraphIROperationCompilable<B>,
{
    fn try_pass_on_node(&self, ir: &mut GraphIR<B>, target: NodeId) -> Result<bool, GraphIRError> {
        let op = ir.get(target)?.op();

        if let Some(Unary { input, op: UnaryOp::DiffableFromOutput(activation) }) = downcast(op) {
            let ir_node = ir.get(input.idx)?;

            if ir_node.children() == 1 {
                if let Some(SparseAffineActivate {
                    weights,
                    biases,
                    values,
                    indices,
                    activation: DiffableFromOutput::Identity,
                }) = downcast(ir_node.op())
                {
                    ir.replace(target, SparseAffineActivate { weights, biases, values, indices, activation })?;
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

#[derive(Debug)]
pub struct LowPriorityFusions;

impl<B: BackendMarker> GraphIRSimplePass<B> for LowPriorityFusions {
    fn try_pass_on_node(&self, ir: &mut GraphIR<B>, target: NodeId) -> Result<bool, GraphIRError> {
        let op = ir.get(target)?.op();

        if let Some(LinearCombination { items, shape }) = downcast(op) {
            if let [(a, alpha), (b, beta)] = &items[..] {
                let a = AnnotatedNode { idx: *a, shape };
                let b = AnnotatedNode { idx: *b, shape };

                if *alpha == 1.0 && *beta == 1.0 {
                    if add_single_dense(ir, a, b, target)? {
                        return Ok(true);
                    }

                    if add_single_dense(ir, b, a, target)? {
                        return Ok(true);
                    }
                }

                if linear_comb_single(ir, *alpha, a, *beta, b, target)? {
                    return Ok(true);
                }

                if linear_comb_single(ir, *beta, b, *alpha, a, target)? {
                    return Ok(true);
                }
            }

            return compact_linear_comb(ir, target);
        }

        if let Some(Unary { input, op: UnaryOp::AbsPow(x) }) = downcast(op) {
            return power_error(ir, input, x, target);
        }

        if let Some(Unary { input, op: UnaryOp::Mul(x) }) = downcast(op) {
            return scale(ir, input, x, target);
        }

        Ok(false)
    }
}

fn power_error<B: BackendMarker>(
    ir: &mut GraphIR<B>,
    node: AnnotatedNode,
    power: f32,
    target: NodeId,
) -> Result<bool, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.children() == 1 {
        if let Some(LinearCombination { items, shape }) = downcast(ir_node.op()) {
            if let [(a, 1.0), (b, -1.0)] | [(a, -1.0), (b, 1.0)] = items[..] {
                let a = AnnotatedNode { idx: a, shape };
                let b = AnnotatedNode { idx: b, shape };

                if a.idx != b.idx && ir.get(a.idx)?.ty().batched == ir.get(b.idx)?.ty().batched {
                    ir.replace(target, AbsPowerError { a, b, power })?;
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

fn scale<B: BackendMarker>(
    ir: &mut GraphIR<B>,
    node: AnnotatedNode,
    scale: f32,
    target: NodeId,
) -> Result<bool, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.children() == 1 {
        if let Some(LinearCombination { items, shape }) = downcast(ir_node.op()) {
            if let [(a, alpha), (b, beta)] = items[..] {
                let a = AnnotatedNode { idx: a, shape };
                let b = AnnotatedNode { idx: b, shape };

                ir.replace(target, LinearCombination::new([(a, alpha * scale), (b, beta * scale)])?)?;
                return Ok(true);
            }
        }
    }

    Ok(false)
}

fn add_single_dense<B: BackendMarker>(
    ir: &mut GraphIR<B>,
    lhs: AnnotatedNode,
    rhs: AnnotatedNode,
    target: NodeId,
) -> Result<bool, GraphIRError> {
    let ir_node = ir.get(lhs.idx)?;

    if ir_node.children() == 1 {
        if let Some(Matmul { a, transa: false, b, transb: false }) = downcast(ir_node.op()) {
            if !ir.get(rhs.idx)?.ty().batched {
                ir.replace(target, Affine { weights: a, inputs: b, biases: rhs })?;
                return Ok(true);
            }
        }
    }

    Ok(false)
}

fn linear_comb_single<B: BackendMarker>(
    ir: &mut GraphIR<B>,
    alpha: f32,
    lhs: AnnotatedNode,
    beta: f32,
    rhs: AnnotatedNode,
    target: NodeId,
) -> Result<bool, GraphIRError> {
    let ir_node = ir.get(lhs.idx)?;

    if ir_node.children() == 1 {
        if let Some(Unary { input, op: UnaryOp::Mul(x) }) = downcast(ir_node.op()) {
            ir.replace(target, LinearCombination::new([(input, alpha * x), (rhs, beta)])?)?;
            return Ok(true);
        }
    }

    Ok(false)
}

fn compact_linear_comb<B: BackendMarker>(ir: &mut GraphIR<B>, target: NodeId) -> Result<bool, GraphIRError> {
    let LinearCombination { items, shape } = downcast(ir.get(target)?.op()).unwrap();

    for &(node, weight) in &items {
        let ir_node = ir.get(node)?;

        if let Some(LinearCombination { items: par_items, shape: shape2 }) = downcast(ir_node.op()) {
            assert_eq!(shape.size(), shape2.size());

            let mut items: Vec<_> = items
                .iter()
                .copied()
                .filter_map(|(idx, weight)| (idx != node).then_some((AnnotatedNode { idx, shape }, weight)))
                .collect();

            for (par_node, par_weight) in par_items {
                items.push((AnnotatedNode { idx: par_node, shape }, par_weight * weight));
            }

            ir.replace(target, LinearCombination::new(items)?)?;
            return Ok(true);
        }
    }

    Ok(false)
}
