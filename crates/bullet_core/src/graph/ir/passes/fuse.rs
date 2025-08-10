use crate::graph::ir::{
    node::AnnotatedNode,
    operation::{
        affine::{Affine, Matmul},
        binary::AbsPowerError,
        nary::LinearCombination,
        sparse::SparseAffineActivate,
        unary::{DiffableFromOutput, Unary, UnaryOp},
    },
    transform::GraphIRTransform,
    BackendMarker, GraphIR, GraphIRError, GraphIRNode,
};

use super::downcast;

pub fn diffable_from_output<B: BackendMarker>(
    ir: &GraphIR<B>,
    node: AnnotatedNode,
    activation: DiffableFromOutput,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
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
                return GraphIRTransform::new([node.idx], vec![new_data]);
            }
        }
    }

    Ok(None)
}

pub fn power_error<B: BackendMarker>(
    ir: &GraphIR<B>,
    node: AnnotatedNode,
    power: f32,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(LinearCombination { items, shape }) = downcast(op) {
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
    }

    Ok(None)
}

pub fn scale<B: BackendMarker>(
    ir: &GraphIR<B>,
    node: AnnotatedNode,
    scale: f32,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(LinearCombination { items, shape }) = downcast(op) {
                if let [(a, alpha), (b, beta)] = items[..] {
                    let a = AnnotatedNode { idx: a, shape: *shape };
                    let b = AnnotatedNode { idx: b, shape: *shape };

                    let new_data =
                        old_data.with_new_op(LinearCombination::new([(a, alpha * scale), (b, beta * scale)])?);
                    return GraphIRTransform::new([node.idx], [new_data]);
                }
            }
        }
    }

    Ok(None)
}

pub fn add_single_sparse<B: BackendMarker>(
    ir: &GraphIR<B>,
    lhs: AnnotatedNode,
    rhs: AnnotatedNode,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(lhs.idx)?;

    if ir_node.num_children == 1 {
        if let Some(Some(&SparseAffineActivate {
            weights,
            indices,
            values,
            biases: None,
            activation: DiffableFromOutput::Identity,
        })) = ir_node.parent_operation.as_ref().map(downcast)
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

pub fn add_single_dense<B: BackendMarker>(
    ir: &GraphIR<B>,
    lhs: AnnotatedNode,
    rhs: AnnotatedNode,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(lhs.idx)?;

    if ir_node.num_children == 1 {
        if let Some(Some(&Matmul { a, transa: false, b, transb: false })) =
            ir_node.parent_operation.as_ref().map(downcast)
        {
            if !ir.get(rhs.idx)?.info.batched {
                let new_data = old_data.with_new_op(Affine { weights: a, inputs: b, biases: rhs });
                return GraphIRTransform::new([lhs.idx], [new_data]);
            }
        }
    }

    Ok(None)
}

pub fn linear_comb_single<B: BackendMarker>(
    ir: &GraphIR<B>,
    alpha: f32,
    lhs: AnnotatedNode,
    beta: f32,
    rhs: AnnotatedNode,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(lhs.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&Unary { input, op: UnaryOp::Mul(x) }) = downcast(op) {
                let new_data = old_data.with_new_op(LinearCombination::new([(input, alpha * x), (rhs, beta)])?);
                return GraphIRTransform::new([lhs.idx], vec![new_data]);
            }
        }
    }

    Ok(None)
}

pub fn compact_linear_comb<B: BackendMarker>(
    ir: &GraphIR<B>,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let LinearCombination { items, shape } = downcast(old_data.parent_operation.as_ref().unwrap()).unwrap();
    let shape = *shape;

    for &(node, weight) in items {
        let ir_node = ir.get(node)?;

        if let Some(Some(LinearCombination { items: par_items, shape: shape2 })) =
            ir_node.parent_operation.as_ref().map(downcast)
        {
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
