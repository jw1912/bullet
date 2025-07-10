use crate::graph::ir::operation::sparse::SparseAffineDualActivate;

use super::{
    node::AnnotatedNode,
    operation::{
        affine::{Affine, Matmul},
        binary::Concat,
        binary::{Binary, BinaryOp},
        sparse::SparseAffineActivate,
        unary::PairwiseMul,
        unary::{DiffableFromOutput, Unary, UnaryOp},
        GraphIROperation,
    },
    GraphIR, GraphIRError, GraphIRNode,
};

pub struct FusionDescription {
    pub eliminated: Vec<usize>,
    pub new_nodes: Vec<GraphIRNode>,
}

impl FusionDescription {
    pub fn new(
        eliminated: &[usize],
        new_nodes: impl Into<Vec<GraphIRNode>>,
    ) -> Result<Option<FusionDescription>, GraphIRError> {
        Ok(Some(FusionDescription { eliminated: eliminated.to_vec(), new_nodes: new_nodes.into() }))
    }
}

pub fn search_for_fusion(ir: &GraphIR, node: usize) -> Result<Option<FusionDescription>, GraphIRError> {
    let data = ir.get(node).unwrap();

    if let Some(op) = &data.parent_operation {
        if let Some(Binary { a, b, op: BinaryOp::LinearCombination { alpha, beta } }) = downcast(op) {
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

fn fuse_diffable_from_output(
    ir: &GraphIR,
    node: &AnnotatedNode,
    activation: DiffableFromOutput,
    old_data: &GraphIRNode,
) -> Result<Option<FusionDescription>, GraphIRError> {
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

fn fuse_concat(
    ir: &GraphIR,
    a: &AnnotatedNode,
    b: &AnnotatedNode,
    old_data: &GraphIRNode,
) -> Result<Option<FusionDescription>, GraphIRError> {
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
                        shape,
                        parent_operation: Some(Box::new(op)),
                        requires_grad,
                        num_children: 0,
                        idx: new_b.idx,
                        sparse: None,
                        batched,
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
                        shape,
                        parent_operation: Some(Box::new(op)),
                        requires_grad,
                        num_children: 0,
                        idx: new_b.idx,
                        sparse: None,
                        batched,
                    };

                    let new_op = old_data.with_new_op(Unary { input: new_b, op: op1 });
                    return FusionDescription::new(&[a.idx, b.idx], vec![new_concat, new_op]);
                }
            }
        }
    }

    Ok(None)
}

fn fuse_power_error(
    ir: &GraphIR,
    node: &AnnotatedNode,
    power: f32,
    old_data: &GraphIRNode,
) -> Result<Option<FusionDescription>, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&Binary { a, b, op: BinaryOp::LinearCombination { alpha: 1.0, beta: -1.0 } }) = downcast(op) {
                if a.idx != b.idx && ir.get(a.idx)?.batched == ir.get(b.idx)?.batched {
                    let new_data = old_data.with_new_op(Binary { a, b, op: BinaryOp::PowerError { power } });
                    return FusionDescription::new(&[node.idx], [new_data]);
                }
            }
        }
    }

    Ok(None)
}

fn fuse_scale(
    ir: &GraphIR,
    node: &AnnotatedNode,
    scale: f32,
    old_data: &GraphIRNode,
) -> Result<Option<FusionDescription>, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&Binary { a, b, op: BinaryOp::LinearCombination { alpha, beta } }) = downcast(op) {
                let new_data = old_data.with_new_op(Binary {
                    a,
                    b,
                    op: BinaryOp::LinearCombination { alpha: alpha * scale, beta: beta * scale },
                });
                return FusionDescription::new(&[node.idx], [new_data]);
            }
        }
    }

    Ok(None)
}

fn fuse_linear_comb(
    ir: &GraphIR,
    alpha: f32,
    lhs: &AnnotatedNode,
    beta: f32,
    rhs: &AnnotatedNode,
    data: &GraphIRNode,
) -> Result<Option<FusionDescription>, GraphIRError> {
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

fn fuse_add_single(
    ir: &GraphIR,
    lhs: &AnnotatedNode,
    rhs: &AnnotatedNode,
    old_data: &GraphIRNode,
) -> Result<Option<FusionDescription>, GraphIRError> {
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
                if !ir.get(rhs.idx)?.batched {
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

fn fuse_linear_comb_single(
    ir: &GraphIR,
    alpha: f32,
    lhs: &AnnotatedNode,
    beta: f32,
    rhs: &AnnotatedNode,
    old_data: &GraphIRNode,
) -> Result<Option<FusionDescription>, GraphIRError> {
    let ir_node = ir.get(lhs.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&Unary { input, op: UnaryOp::Mul(x) }) = downcast(op) {
                let new_data = old_data.with_new_op(Binary {
                    a: input,
                    b: *rhs,
                    op: BinaryOp::LinearCombination { alpha: alpha * x, beta },
                });
                return FusionDescription::new(&[lhs.idx], vec![new_data]);
            }
        }
    }

    Ok(None)
}

#[allow(clippy::borrowed_box)]
fn downcast<T: 'static>(op: &Box<dyn GraphIROperation>) -> Option<&T> {
    let op: &dyn std::any::Any = op.as_ref();
    op.downcast_ref()
}
