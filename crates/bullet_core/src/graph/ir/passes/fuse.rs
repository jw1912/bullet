use crate::graph::ir::{
    node::AnnotatedNode,
    operation::{
        affine::{Affine, Matmul},
        binary::{AbsPowerError, LinearCombination},
        sparse::SparseAffineActivate,
        unary::{DiffableFromOutput, Unary, UnaryOp},
    },
    transform::GraphIRTransform,
    BackendMarker, GraphIR, GraphIRError, GraphIRNode,
};

use super::downcast;

pub fn diffable_from_output<B: BackendMarker>(
    ir: &GraphIR<B>,
    node: &AnnotatedNode,
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
    node: &AnnotatedNode,
    power: f32,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&LinearCombination { a, b, alpha: 1.0, beta: -1.0 }) = downcast(op) {
                if a.idx != b.idx && ir.get(a.idx)?.info.batched == ir.get(b.idx)?.info.batched {
                    let new_data = old_data.with_new_op(AbsPowerError { a, b, power });
                    return GraphIRTransform::new([node.idx], [new_data]);
                }
            }
        }
    }

    Ok(None)
}

pub fn scale<B: BackendMarker>(
    ir: &GraphIR<B>,
    node: &AnnotatedNode,
    scale: f32,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&LinearCombination { a, b, alpha, beta }) = downcast(op) {
                let new_data =
                    old_data.with_new_op(LinearCombination { a, b, alpha: alpha * scale, beta: beta * scale });
                return GraphIRTransform::new([node.idx], [new_data]);
            }
        }
    }

    Ok(None)
}

pub fn linear_comb<B: BackendMarker>(
    ir: &GraphIR<B>,
    alpha: f32,
    lhs: &AnnotatedNode,
    beta: f32,
    rhs: &AnnotatedNode,
    data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    if alpha == 1.0 && beta == 1.0 {
        if let Some(fusion_data) = add_single(ir, lhs, rhs, data)? {
            return Ok(Some(fusion_data));
        }

        if let Some(fusion_data) = add_single(ir, rhs, lhs, data)? {
            return Ok(Some(fusion_data));
        }
    }

    if let Some(fusion_data) = linear_comb_single(ir, alpha, lhs, beta, rhs, data)? {
        return Ok(Some(fusion_data));
    }

    if let Some(fusion_data) = linear_comb_single(ir, beta, rhs, alpha, lhs, data)? {
        return Ok(Some(fusion_data));
    }

    Ok(None)
}

pub fn add_single<B: BackendMarker>(
    ir: &GraphIR<B>,
    lhs: &AnnotatedNode,
    rhs: &AnnotatedNode,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
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
                return GraphIRTransform::new([lhs.idx], [new_data]);
            }

            if let Some(&Matmul { a, transa: false, b, transb: false }) = downcast(op) {
                if !ir.get(rhs.idx)?.info.batched {
                    let new_data = old_data.with_new_op(Affine { weights: a, inputs: b, biases: *rhs });
                    return GraphIRTransform::new([lhs.idx], [new_data]);
                }
            }
        }
    }

    Ok(None)
}

pub fn linear_comb_single<B: BackendMarker>(
    ir: &GraphIR<B>,
    alpha: f32,
    lhs: &AnnotatedNode,
    beta: f32,
    rhs: &AnnotatedNode,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(lhs.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&Unary { input, op: UnaryOp::Mul(x) }) = downcast(op) {
                let new_data = old_data.with_new_op(LinearCombination { a: input, b: *rhs, alpha: alpha * x, beta });
                return GraphIRTransform::new([lhs.idx], vec![new_data]);
            }
        }
    }

    Ok(None)
}
