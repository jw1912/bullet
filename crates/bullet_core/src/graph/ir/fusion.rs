use crate::{backend::device::base::Activation, graph::ir::node::AnnotatedNode};

use super::{op::GraphIROp, GraphIR, GraphIRNode};

use GraphIROp::*;

pub fn fusion_pass(ir: &mut GraphIR, node: usize) -> bool {
    if let Some((eliminated, new_data)) = search_for_fusion(ir, node) {
        for dead in eliminated {
            if let Some(Some(op)) = ir.nodes[dead].as_ref().map(|x| x.parent_operation) {
                for parent in op.nodes() {
                    ir.nodes[parent.idx].as_mut().unwrap().num_children -= 1;
                }
            }

            ir.nodes[dead] = None;
        }

        ir.nodes[node] = Some(new_data);

        true
    } else {
        false
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
            Activate(node, act) => return fuse_activation(ir, node, *act, data),
            _ => {}
        }
    }

    None
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
                SparseAffineActivate(weights, input, None, Activation::Identity) => {
                    new_data.parent_operation =
                        Some(SparseAffineActivate(weights, input, Some(*rhs), Activation::Identity));
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

fn fuse_activation(
    ir: &GraphIR,
    node: &AnnotatedNode,
    activation: Activation,
    old_data: &GraphIRNode,
) -> Option<(Vec<usize>, GraphIRNode)> {
    let ir_node = get_ir_node(ir, node);

    if activation == Activation::Square {
        return None;
    }

    if ir_node.num_children == 1 {
        if let Some(op) = ir_node.parent_operation {
            let mut new_data = old_data.clone();

            match op {
                SparseAffineActivate(a, b, c, Activation::Identity) => {
                    new_data.parent_operation = Some(SparseAffineActivate(a, b, c, activation));
                    return Some((vec![node.idx], new_data));
                }
                SparseAffineDualActivate(w, n, s, b, Activation::Identity) => {
                    new_data.parent_operation = Some(SparseAffineDualActivate(w, n, s, b, activation));
                    return Some((vec![node.idx], new_data));
                }
                _ => {}
            }
        }
    }

    None
}
