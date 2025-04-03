use super::{
    node::AnnotatedNode,
    op::{DiffableFromOutput, GraphIROp, UnaryOp},
    GraphIR, GraphIRError, GraphIRNode,
};

use GraphIROp::*;

pub struct FusionDescription {
    pub eliminated: Vec<usize>,
    pub new_nodes: Vec<GraphIRNode>,
}

impl FusionDescription {
    pub fn new(eliminated: &[usize], new_nodes: &[GraphIRNode]) -> Result<Option<FusionDescription>, GraphIRError> {
        Ok(Some(FusionDescription { eliminated: eliminated.to_vec(), new_nodes: new_nodes.to_vec() }))
    }
}

pub fn search_for_fusion(ir: &GraphIR, node: usize) -> Result<Option<FusionDescription>, GraphIRError> {
    let data = ir.get(node).unwrap();

    if let Some(op) = data.parent_operation.as_ref() {
        match op {
            LinearCombination(alpha, lhs, beta, rhs) => return fuse_linear_comb(ir, *alpha, lhs, *beta, rhs, data),
            Concat(a, b) => return fuse_concat(ir, a, b, data),
            Unary(node, UnaryOp::DiffableFromOutput(act)) => return fuse_diffable_from_output(ir, node, *act, data),
            Unary(node, UnaryOp::AbsPow(x)) => return fuse_power_error(ir, node, *x, data),
            Unary(node, UnaryOp::Mul(x)) => return fuse_scale(ir, node, *x, data),
            _ => {}
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
        if let Some(op) = ir_node.parent_operation {
            let mut new_data = old_data.clone();

            match op {
                SparseAffineActivate(a, b, c, DiffableFromOutput::Identity) => {
                    new_data.parent_operation = Some(SparseAffineActivate(a, b, c, activation));
                    return FusionDescription::new(&[node.idx], &[new_data]);
                }
                SparseAffineDualActivate(w, n, s, b, DiffableFromOutput::Identity) => {
                    new_data.parent_operation = Some(SparseAffineDualActivate(w, n, s, b, activation));
                    return FusionDescription::new(&[node.idx], &[new_data]);
                }
                _ => {}
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
        match (node_a.parent_operation, node_b.parent_operation) {
            (Some(SparseAffineActivate(wa, ia, ba, acta)), Some(SparseAffineActivate(wb, ib, bb, actb))) => {
                if wa == wb && ia.idx != ib.idx && ba == bb && acta == actb {
                    let mut new_data = old_data.clone();
                    new_data.parent_operation = Some(SparseAffineDualActivate(wa, ia, ib, ba, acta));
                    return FusionDescription::new(&[a.idx, b.idx], &[new_data]);
                }
            }
            (Some(PairwiseMul(c, false)), Some(PairwiseMul(d, false))) => {
                if c.idx != d.idx && c.shape.size() == d.shape.size() {
                    let op = Concat(c, d);

                    let (shape, batched, requires_grad) = op.output_info(ir)?;
                    let new_b = AnnotatedNode { idx: node_b.idx, shape };

                    let new_concat = GraphIRNode {
                        id: node_b.id.clone(),
                        shape,
                        parent_operation: Some(op),
                        requires_grad,
                        num_children: 0,
                        idx: new_b.idx,
                        sparse: None,
                        batched,
                    };

                    let mut new_pairwise = old_data.clone();
                    new_pairwise.parent_operation = Some(PairwiseMul(new_b, true));
                    return FusionDescription::new(&[a.idx, b.idx], &[new_concat, new_pairwise]);
                }
            }
            (Some(Unary(c, op1)), Some(Unary(d, op2))) => {
                if op1 == op2 && c.idx != d.idx && c.shape.size() == d.shape.size() {
                    let op = Concat(c, d);

                    let (shape, batched, requires_grad) = op.output_info(ir)?;
                    let new_b = AnnotatedNode { idx: node_b.idx, shape };

                    let new_concat = GraphIRNode {
                        id: node_b.id.clone(),
                        shape,
                        parent_operation: Some(op),
                        requires_grad,
                        num_children: 0,
                        idx: new_b.idx,
                        sparse: None,
                        batched,
                    };

                    let mut new_op = old_data.clone();
                    new_op.parent_operation = Some(Unary(new_b, op1));
                    return FusionDescription::new(&[a.idx, b.idx], &[new_concat, new_op]);
                }
            }
            _ => {}
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
        if let Some(LinearCombination(1.0, a, -1.0, b)) = ir_node.parent_operation {
            if a.idx != b.idx && ir.get(a.idx)?.batched == ir.get(b.idx)?.batched {
                let mut new_data = old_data.clone();
                new_data.parent_operation = Some(PowerError(a, b, power));
                return FusionDescription::new(&[node.idx], &[new_data]);
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
        if let Some(LinearCombination(alpha, a, beta, b)) = ir_node.parent_operation {
            let mut new_data = old_data.clone();
            new_data.parent_operation = Some(LinearCombination(alpha * scale, a, beta * scale, b));
            return FusionDescription::new(&[node.idx], &[new_data]);
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
        if let Some(op) = ir_node.parent_operation {
            let mut new_data = old_data.clone();

            match op {
                SparseAffineActivate(weights, input, None, DiffableFromOutput::Identity) => {
                    new_data.parent_operation =
                        Some(SparseAffineActivate(weights, input, Some(*rhs), DiffableFromOutput::Identity));
                    return FusionDescription::new(&[lhs.idx], &[new_data]);
                }
                SparseAffineDualActivate(weights, s, n, None, DiffableFromOutput::Identity) => {
                    new_data.parent_operation =
                        Some(SparseAffineDualActivate(weights, s, n, Some(*rhs), DiffableFromOutput::Identity));
                    return FusionDescription::new(&[lhs.idx], &[new_data]);
                }
                Matmul(a, false, b, false) => {
                    if !ir.get(rhs.idx)?.batched {
                        new_data.parent_operation = Some(Affine(a, b, *rhs));
                        return FusionDescription::new(&[lhs.idx], &[new_data]);
                    }
                }
                _ => {}
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
        if let Some(Unary(node, UnaryOp::Mul(x))) = ir_node.parent_operation {
            let mut new_data = old_data.clone();
            new_data.parent_operation = Some(LinearCombination(alpha * x, node, beta, *rhs));
            return FusionDescription::new(&[lhs.idx], &[new_data]);
        }
    }

    Ok(None)
}
