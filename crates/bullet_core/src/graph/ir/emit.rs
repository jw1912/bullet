use std::{collections::HashMap, fmt};

use crate::{
    backend::device::blas::Shape,
    graph::ir::{node::AnnotatedNode, op::UnaryOp},
};

use super::{GraphIR, GraphIRNode};

impl fmt::Display for GraphIR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut orig_shapes = HashMap::new();

        for node in self.nodes.iter().flatten() {
            writeln!(f, "%{:0>2} = {};", node.own.idx, op_name_only(node, &mut orig_shapes))?;
        }

        Ok(())
    }
}

fn op_name_only(node: &GraphIRNode, shapes: &mut HashMap<usize, Shape>) -> String {
    use super::op::GraphIROp::*;

    shapes.insert(node.own.idx, node.own.shape);

    let id = |x: &AnnotatedNode| {
        if *shapes.get(&x.idx).unwrap() == x.shape {
            format!("%{}", x.idx)
        } else {
            format!("%{}: {}", x.idx, x.shape)
        }
    };

    match node.parent_operation.as_ref() {
        Some(op) => match op {
            Affine(a, b, c) => format!("Affine({}, {}, {})", id(a), id(b), id(c)),
            Concat(a, b) => format!("Concat({}, {})", id(a), id(b)),
            Gather(input, mask) => format!("Gather({}, {})", id(input), id(mask)),
            LinearCombination(alpha, a, beta, b) => format!("LinearCombination({alpha}, {}, {beta}, {})", id(a), id(b)),
            Mask(input, mask) => format!("Mask({}, {})", id(input), id(mask)),
            Matmul(a, ta, b, tb) => format!("Matmul({}, {ta}, {}, {tb})", id(a), id(b)),
            PairwiseMul(input, post_concat) => format!("PairwiseMul({}, {post_concat})", id(input)),
            PowerError(a, b, pow) => format!("PowerError({}, {}, {pow})", id(a), id(b)),
            ReduceAcrossBatch(node) => format!("ReduceAcrossBatch({})", id(node)),
            Select(input, buckets) => format!("Select({}, {})", id(input), id(buckets)),
            Slice(input, a, b) => format!("Slice({}, {a}, {b})", id(input)),
            SparseAffineActivate(w, i, b, act) => {
                let bias = if let Some(b) = b { id(b) } else { "None".to_string() };
                format!("SparseAffineActivate({}, {}, {bias}, {act:?})", id(w), id(i))
            }
            ToDense(node) => format!("ToDense({})", id(node)),
            Unary(node, unary) => match unary {
                UnaryOp::DiffableFromOutput(act) => format!("{act:?}({})", id(node)),
                UnaryOp::Add(x) => format!("Add({}, {x})", id(node)),
                UnaryOp::Mul(x) => format!("Mul({}, {x})", id(node)),
                UnaryOp::AbsPow(x) => format!("AbsPow({}, {x})", id(node)),
            },
            SparseAffineDualActivate(w, s, n, b, act) => {
                format!("SparseAffineDualActivate({}, {}, {}, {}, {act:?})", id(w), id(s), id(n), id(b))
            }
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => {
                format!("MaskedSoftmaxCrossEntropyLoss({}, {}, {})", id(mask), id(input), id(target))
            }
            SoftmaxCrossEntropyLoss(a, b) => format!("SoftmaxCrossEntropyLoss({}, {})", id(a), id(b)),
        },
        None => {
            let layout = match node.own.sparse {
                Some(nnz) => format!("Sparse(f32, {nnz})"),
                None => "Dense(f32)".to_string(),
            };
            format!("Create({}, {}, {}, {})", node.own.shape, layout, node.requires_grad, node.own.can_be_batched)
        }
    }
}
