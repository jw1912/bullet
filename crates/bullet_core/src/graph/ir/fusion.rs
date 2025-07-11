use crate::graph::ir::{
    node::NodeInfo,
    operation::{sparse::SparseAffineDualActivate, GraphIROperationCompilable},
    BackendMarker,
};

use super::{
    node::AnnotatedNode,
    operation::{
        affine::{Affine, Matmul},
        binary::Concat,
        binary::{AbsPowerError, LinearCombination},
        sparse::SparseAffineActivate,
        unary::PairwiseMul,
        unary::{DiffableFromOutput, Unary, UnaryOp},
        GraphIROperation,
    },
    GraphIR, GraphIRError, GraphIRNode,
};

pub struct FusionDescription<B: BackendMarker> {
    pub eliminated: Vec<usize>,
    pub new_nodes: Vec<GraphIRNode<B>>,
}

impl<B: BackendMarker> FusionDescription<B> {
    pub fn new(
        eliminated: &[usize],
        new_nodes: impl Into<Vec<GraphIRNode<B>>>,
    ) -> Result<Option<FusionDescription<B>>, GraphIRError> {
        Ok(Some(FusionDescription { eliminated: eliminated.to_vec(), new_nodes: new_nodes.into() }))
    }
}

pub fn search_for_fusion<B: BackendMarker>(
    ir: &GraphIR<B>,
    node: usize,
) -> Result<Option<FusionDescription<B>>, GraphIRError> {
    let data = ir.get(node).unwrap();

    if let Some(op) = &data.parent_operation {
        if let Some(LinearCombination { a, b, alpha, beta }) = downcast(op) {
            return fuse_linear_comb(ir, *alpha, a, *beta, b, data);
        }

        if let Some(Concat { a, b }) = downcast(op) {
            return fuse_concat(ir, a, b, data);
        }

        if let Some(Unary { input, op: UnaryOp::DiffableFromOutput(act) }) = downcast(op) {
            return fuse_diffable_from_output(ir, input, *act, data);
        }

        if let Some(Unary { input, op: UnaryOp::AbsPow(x) }) = downcast(op) {
            return fuse_power_error(ir, input, *x, data);
        }

        if let Some(Unary { input, op: UnaryOp::Mul(x) }) = downcast(op) {
            return fuse_scale(ir, input, *x, data);
        }
    }

    Ok(None)
}

fn fuse_diffable_from_output<B: BackendMarker>(
    ir: &GraphIR<B>,
    node: &AnnotatedNode,
    activation: DiffableFromOutput,
    old_data: &GraphIRNode<B>,
) -> Result<Option<FusionDescription<B>>, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&SparseAffineActivate {
                weights,
                biases,
                values,
                indices,
                activation: DiffableFromOutput::Identity,
            }) = downcast(op)
            {
                let new_data =
                    old_data.with_new_op(SparseAffineActivate { weights, biases, values, indices, activation });
                return FusionDescription::new(&[node.idx], vec![new_data]);
            }

            if let Some(&SparseAffineDualActivate {
                weights,
                indices_l,
                indices_r,
                biases,
                activation: DiffableFromOutput::Identity,
            }) = downcast(op)
            {
                let new_data = old_data.with_new_op(SparseAffineDualActivate {
                    weights,
                    indices_l,
                    indices_r,
                    biases,
                    activation,
                });
                return FusionDescription::new(&[node.idx], vec![new_data]);
            }
        }
    }

    Ok(None)
}

fn fuse_concat<B: BackendMarker>(
    ir: &GraphIR<B>,
    a: &AnnotatedNode,
    b: &AnnotatedNode,
    old_data: &GraphIRNode<B>,
) -> Result<Option<FusionDescription<B>>, GraphIRError> {
    let node_a = ir.get(a.idx)?;
    let node_b = ir.get(b.idx)?;

    if node_a.num_children == 1 && node_b.num_children == 1 {
        if let (Some(op1), Some(op2)) = (&node_a.parent_operation, &node_b.parent_operation) {
            if let (
                Some(&SparseAffineActivate { weights: wa, indices: ia, values: None, biases: ba, activation: acta }),
                Some(&SparseAffineActivate { weights: wb, indices: ib, values: None, biases: bb, activation: actb }),
            ) = (downcast(op1), downcast(op2))
            {
                if wa == wb && ia.idx != ib.idx && ba == bb && acta == actb {
                    let new_data = old_data.with_new_op(SparseAffineDualActivate {
                        weights: wa,
                        indices_l: ia,
                        indices_r: ib,
                        biases: ba,
                        activation: acta,
                    });
                    return FusionDescription::new(&[a.idx, b.idx], vec![new_data]);
                }
            }

            if let (
                Some(&PairwiseMul { input: c, post_concat: false }),
                Some(&PairwiseMul { input: d, post_concat: false }),
            ) = (downcast(op1), downcast(op2))
            {
                if c.idx != d.idx && c.shape.size() == d.shape.size() {
                    let op = Concat { a: c, b: d };

                    let shape = op.output_shape(ir)?;
                    let batched = op.output_batched(ir)?;
                    let requires_grad = op.output_requires_grad(ir)?;
                    let new_b = AnnotatedNode { idx: node_b.idx, shape };

                    let new_concat = GraphIRNode {
                        id: node_b.id.clone(),
                        parent_operation: Some(Box::new(op)),
                        num_children: 0,
                        idx: new_b.idx,
                        info: NodeInfo { shape, requires_grad, sparse: None, batched },
                    };

                    let new_pairwise = old_data.with_new_op(PairwiseMul { input: new_b, post_concat: true });
                    return FusionDescription::new(&[a.idx, b.idx], vec![new_concat, new_pairwise]);
                }
            }

            if let (Some(&Unary { input: c, op: op1 }), Some(&Unary { input: d, op: op2 })) =
                (downcast(op1), downcast(op2))
            {
                if op1 == op2 && c.idx != d.idx && c.shape.size() == d.shape.size() {
                    let op = Concat { a: c, b: d };

                    let shape = op.output_shape(ir)?;
                    let batched = op.output_batched(ir)?;
                    let requires_grad = op.output_requires_grad(ir)?;
                    let new_b = AnnotatedNode { idx: node_b.idx, shape };

                    let new_concat = GraphIRNode {
                        id: node_b.id.clone(),
                        parent_operation: Some(Box::new(op)),
                        num_children: 0,
                        idx: new_b.idx,
                        info: NodeInfo { shape, requires_grad, sparse: None, batched },
                    };

                    let new_op = old_data.with_new_op(Unary { input: new_b, op: op1 });
                    return FusionDescription::new(&[a.idx, b.idx], vec![new_concat, new_op]);
                }
            }
        }
    }

    Ok(None)
}

fn fuse_power_error<B: BackendMarker>(
    ir: &GraphIR<B>,
    node: &AnnotatedNode,
    power: f32,
    old_data: &GraphIRNode<B>,
) -> Result<Option<FusionDescription<B>>, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&LinearCombination { a, b, alpha: 1.0, beta: -1.0 }) = downcast(op) {
                if a.idx != b.idx && ir.get(a.idx)?.info.batched == ir.get(b.idx)?.info.batched {
                    let new_data = old_data.with_new_op(AbsPowerError { a, b, power });
                    return FusionDescription::new(&[node.idx], [new_data]);
                }
            }
        }
    }

    Ok(None)
}

fn fuse_scale<B: BackendMarker>(
    ir: &GraphIR<B>,
    node: &AnnotatedNode,
    scale: f32,
    old_data: &GraphIRNode<B>,
) -> Result<Option<FusionDescription<B>>, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&LinearCombination { a, b, alpha, beta }) = downcast(op) {
                let new_data =
                    old_data.with_new_op(LinearCombination { a, b, alpha: alpha * scale, beta: beta * scale });
                return FusionDescription::new(&[node.idx], [new_data]);
            }
        }
    }

    Ok(None)
}

fn fuse_linear_comb<B: BackendMarker>(
    ir: &GraphIR<B>,
    alpha: f32,
    lhs: &AnnotatedNode,
    beta: f32,
    rhs: &AnnotatedNode,
    data: &GraphIRNode<B>,
) -> Result<Option<FusionDescription<B>>, GraphIRError> {
    if alpha == 1.0 && beta == 1.0 {
        if let Some(fusion_data) = fuse_add_single(ir, lhs, rhs, data)? {
            return Ok(Some(fusion_data));
        }

        if let Some(fusion_data) = fuse_add_single(ir, rhs, lhs, data)? {
            return Ok(Some(fusion_data));
        }
    }

    if let Some(fusion_data) = fuse_linear_comb_single(ir, alpha, lhs, beta, rhs, data)? {
        return Ok(Some(fusion_data));
    }

    if let Some(fusion_data) = fuse_linear_comb_single(ir, beta, rhs, alpha, lhs, data)? {
        return Ok(Some(fusion_data));
    }

    Ok(None)
}

fn fuse_add_single<B: BackendMarker>(
    ir: &GraphIR<B>,
    lhs: &AnnotatedNode,
    rhs: &AnnotatedNode,
    old_data: &GraphIRNode<B>,
) -> Result<Option<FusionDescription<B>>, GraphIRError> {
    let ir_node = ir.get(lhs.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&SparseAffineActivate {
                weights,
                indices,
                values,
                biases: None,
                activation: DiffableFromOutput::Identity,
            }) = downcast(op)
            {
                let new_data = old_data.with_new_op(SparseAffineActivate {
                    weights,
                    indices,
                    values,
                    biases: Some(*rhs),
                    activation: DiffableFromOutput::Identity,
                });
                return FusionDescription::new(&[lhs.idx], [new_data]);
            }

            if let Some(&Matmul { a, transa: false, b, transb: false }) = downcast(op) {
                if !ir.get(rhs.idx)?.info.batched {
                    let new_data = old_data.with_new_op(Affine { weights: a, inputs: b, biases: *rhs });
                    return FusionDescription::new(&[lhs.idx], [new_data]);
                }
            }

            if let Some(&SparseAffineDualActivate {
                weights,
                indices_l,
                indices_r,
                biases: None,
                activation: DiffableFromOutput::Identity,
            }) = downcast(op)
            {
                let new_data = old_data.with_new_op(SparseAffineDualActivate {
                    weights,
                    indices_l,
                    indices_r,
                    biases: Some(*rhs),
                    activation: DiffableFromOutput::Identity,
                });
                return FusionDescription::new(&[lhs.idx], [new_data]);
            }
        }
    }

    Ok(None)
}

fn fuse_linear_comb_single<B: BackendMarker>(
    ir: &GraphIR<B>,
    alpha: f32,
    lhs: &AnnotatedNode,
    beta: f32,
    rhs: &AnnotatedNode,
    old_data: &GraphIRNode<B>,
) -> Result<Option<FusionDescription<B>>, GraphIRError> {
    let ir_node = ir.get(lhs.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&Unary { input, op: UnaryOp::Mul(x) }) = downcast(op) {
                let new_data = old_data.with_new_op(LinearCombination { a: input, b: *rhs, alpha: alpha * x, beta });
                return FusionDescription::new(&[lhs.idx], vec![new_data]);
            }
        }
    }

    Ok(None)
}

#[allow(clippy::borrowed_box)]
fn downcast<B: BackendMarker, T: 'static>(op: &Box<dyn GraphIROperationCompilable<B>>) -> Option<&T> {
    let op: &dyn std::any::Any = op.as_ref();
    op.downcast_ref()
}
