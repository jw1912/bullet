use crate::{backend::device::base::DiffableFromOutput, graph::ir::node::AnnotatedNode};

use super::{
    op::{GraphIROp, UnaryOp},
    GraphIR, GraphIRNode,
};

use GraphIROp::*;

pub fn fusion_pass(ir: &mut GraphIR, node: usize) -> bool {
    if try_exchange_pairwise_concat(ir, node) {
        return true;
    }

    if let Some((mut eliminated, new_data)) = search_for_fusion(ir, node) {
        assert!(valid(&new_data));
        eliminated.push(node);
        ir.delete_nodes(&eliminated);
        ir.replace_data(node, new_data);
        true
    } else {
        false
    }
}

fn valid(data: &GraphIRNode) -> bool {
    if let Some(op) = data.parent_operation {
        let (shape, batched) = op.output_shape().unwrap();
        data.size == shape.size() && data.own.shape == shape && data.own.can_be_batched == batched
    } else {
        true
    }
}

fn search_for_fusion(ir: &GraphIR, node: usize) -> Option<(Vec<usize>, GraphIRNode)> {
    let data = ir.get(node).unwrap();

    if let Some(op) = data.parent_operation.as_ref() {
        match op {
            LinearCombination(1.0, lhs, 1.0, rhs) => {
                if let Some(fusion_data) = fuse_add(ir, lhs, rhs, data) {
                    return Some(fusion_data);
                }

                if let Some(fusion_data) = fuse_add(ir, rhs, lhs, data) {
                    return Some(fusion_data);
                }
            }
            Concat(a, b) => return fuse_concat(ir, a, b, data),
            Unary(node, UnaryOp::DiffableFromOutput(act)) => return fuse_diffable_from_output(ir, node, *act, data),
            Unary(node, UnaryOp::AbsPow(x)) => return fuse_power_error(ir, node, *x, data),
            _ => {}
        }
    }

    None
}

fn try_exchange_pairwise_concat(ir: &mut GraphIR, node: usize) -> bool {
    let mut data = ir.get(node).unwrap().clone();

    if let Some(Concat(a, b)) = data.parent_operation {
        let an = get_ir_node(ir, &a).clone();
        let bn = get_ir_node(ir, &b).clone();

        if an.num_children == 1 && bn.num_children == 1 {
            if let (Some(PairwiseMul(c, false)), Some(PairwiseMul(d, false))) =
                (&an.parent_operation, &bn.parent_operation)
            {
                if c.idx != d.idx && c.shape.size() == d.shape.size() {
                    ir.delete_nodes(&[a.idx, b.idx, node]);

                    let op = Concat(*c, *d);

                    if let Ok((shape, can_be_batched)) = op.output_shape() {
                        let new_b = AnnotatedNode { idx: bn.own.idx, shape, sparse: None, can_be_batched };

                        let new_data = GraphIRNode {
                            id: bn.id,
                            size: 2 * c.shape.size(),
                            parent_operation: Some(op),
                            requires_grad: bn.requires_grad,
                            num_children: 0,
                            own: new_b,
                        };

                        data.parent_operation = Some(PairwiseMul(new_b, true));

                        assert!(valid(&new_data));
                        assert!(valid(&data), "{data:#?}");

                        ir.replace_data(b.idx, new_data);
                        ir.replace_data(node, data);
                        return true;
                    }
                }
            }
        }
    }

    false
}

fn get_ir_node<'a>(ir: &'a GraphIR, node: &AnnotatedNode) -> &'a GraphIRNode {
    ir.nodes[node.idx].as_ref().unwrap()
}

fn fuse_add(
    ir: &GraphIR,
    lhs: &AnnotatedNode,
    rhs: &AnnotatedNode,
    old_data: &GraphIRNode,
) -> Option<(Vec<usize>, GraphIRNode)> {
    let ir_node = get_ir_node(ir, lhs);

    if ir_node.num_children == 1 {
        if let Some(op) = ir_node.parent_operation {
            let mut new_data = old_data.clone();

            match op {
                SparseAffineActivate(weights, input, None, DiffableFromOutput::Identity) => {
                    new_data.parent_operation =
                        Some(SparseAffineActivate(weights, input, Some(*rhs), DiffableFromOutput::Identity));
                    return Some((vec![lhs.idx], new_data));
                }
                Matmul(a, false, b, false) => {
                    if !rhs.can_be_batched {
                        new_data.parent_operation = Some(Affine(a, b, *rhs));
                        return Some((vec![lhs.idx], new_data));
                    }
                }
                _ => {}
            }
        }
    }

    None
}

fn fuse_diffable_from_output(
    ir: &GraphIR,
    node: &AnnotatedNode,
    activation: DiffableFromOutput,
    old_data: &GraphIRNode,
) -> Option<(Vec<usize>, GraphIRNode)> {
    let ir_node = get_ir_node(ir, node);

    if ir_node.num_children == 1 {
        if let Some(op) = ir_node.parent_operation {
            let mut new_data = old_data.clone();

            match op {
                SparseAffineActivate(a, b, c, DiffableFromOutput::Identity) => {
                    new_data.parent_operation = Some(SparseAffineActivate(a, b, c, activation));
                    return Some((vec![node.idx], new_data));
                }
                SparseAffineDualActivate(w, n, s, b, DiffableFromOutput::Identity) => {
                    new_data.parent_operation = Some(SparseAffineDualActivate(w, n, s, b, activation));
                    return Some((vec![node.idx], new_data));
                }
                _ => {}
            }
        }
    }

    None
}

fn fuse_concat(
    ir: &GraphIR,
    a: &AnnotatedNode,
    b: &AnnotatedNode,
    old_data: &GraphIRNode,
) -> Option<(Vec<usize>, GraphIRNode)> {
    let node_a = get_ir_node(ir, a);
    let node_b = get_ir_node(ir, b);

    if node_a.num_children == 1 && node_b.num_children == 1 && node_a.requires_grad && node_b.requires_grad {
        if let (Some(SparseAffineActivate(wa, ia, ba, acta)), Some(SparseAffineActivate(wb, ib, bb, actb))) =
            (node_a.parent_operation, node_b.parent_operation)
        {
            if let Some(bias) = ba {
                if wa == wb && ia.idx != ib.idx && ba == bb && acta == actb {
                    let mut new_data = old_data.clone();
                    new_data.parent_operation = Some(SparseAffineDualActivate(wa, ia, ib, bias, acta));
                    return Some((vec![a.idx, b.idx], new_data));
                }
            }
        }
    }

    None
}

fn fuse_power_error(
    ir: &GraphIR,
    node: &AnnotatedNode,
    power: f32,
    old_data: &GraphIRNode,
) -> Option<(Vec<usize>, GraphIRNode)> {
    let ir_node = get_ir_node(ir, node);

    if ir_node.num_children == 1 {
        if let Some(op) = ir_node.parent_operation {
            let mut new_data = old_data.clone();

            if let LinearCombination(1.0, a, -1.0, b) = op {
                if a.idx != b.idx && a.can_be_batched == b.can_be_batched {
                    new_data.parent_operation = Some(PowerError(a, b, power));
                    return Some((vec![node.idx], new_data));
                }
            }
        }
    }

    None
}
