use crate::{backend::device::base::Activation, graph::ir::node::AnnotatedNode};

use super::{op::GraphIROp, GraphIR, GraphIRNode};

use GraphIROp::*;

pub fn fusion_pass(ir: &mut GraphIR, node: usize) -> bool {
    if let Some((mut eliminated, new_data)) = search_for_fusion(ir, node) {
        eliminated.push(node);
        delete_eliminated(ir, &eliminated);
        add_new(ir, node, new_data);
        true
    } else {
        false
    }
}

fn delete_eliminated(ir: &mut GraphIR, eliminated: &[usize]) {
    for &dead in eliminated {
        if let Some(Some(op)) = ir.nodes[dead].as_ref().map(|x| x.parent_operation) {
            for parent in op.nodes() {
                ir.nodes[parent.idx].as_mut().unwrap().num_children -= 1;
            }
        }
    }

    for &dead in eliminated {
        ir.nodes[dead] = None;
    }
}

fn add_new(ir: &mut GraphIR, node: usize, data: GraphIRNode) {
    for parent in data.parent_operation.as_ref().unwrap().nodes() {
        ir.nodes[parent.idx].as_mut().unwrap().num_children += 1;
    }

    ir.nodes[node] = Some(data);
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

fn fuse_concat(
    ir: &GraphIR,
    a: &AnnotatedNode,
    b: &AnnotatedNode,
    old_data: &GraphIRNode,
) -> Option<(Vec<usize>, GraphIRNode)> {
    let node_a = get_ir_node(ir, a);
    let node_b = get_ir_node(ir, b);

    if node_a.num_children == 1 && node_b.num_children == 1 {
        if let (Some(SparseAffineActivate(wa, ia, ba, acta)), Some(SparseAffineActivate(wb, ib, bb, actb))) =
            (node_a.parent_operation, node_b.parent_operation)
        {
            if let Some(bias) = ba {
                if wa == wb && ia != ib && ba == bb && acta == actb {
                    let mut new_data = old_data.clone();
                    new_data.parent_operation = Some(SparseAffineDualActivate(wa, ia, ib, bias, acta));
                    return Some((vec![a.idx, b.idx], new_data));
                }
            }
        }
    }

    None
}
