use crate::graph::{
    builder::Shape,
    ir::{
        node::{AnnotatedNode, NodeInfo},
        operation::{
            affine::{Affine, Matmul},
            binary::{AbsPowerError, Concat, LinearCombination},
            sparse::SparseAffineActivate,
            unary::{DiffableFromOutput, Slice, Unary, UnaryOp},
            GraphIROperationCompilable,
        },
        transform::GraphIRTransform,
        BackendMarker, GraphIR, GraphIRError, GraphIRNode,
    },
};

pub fn search_for_fusion<B: BackendMarker>(
    ir: &GraphIR<B>,
    node: usize,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let data = ir.get(node)?;

    if let Some(op) = &data.parent_operation {
        if let Some(Matmul { a, b, transa: false, transb: false }) = downcast(op) {
            return exchange_matmul_concat(ir, *a, *b, data);
        }

        if let Some(LinearCombination { a, b, alpha, beta }) = downcast(op) {
            return fuse_linear_comb(ir, *alpha, a, *beta, b, data);
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

fn exchange_matmul_concat<B: BackendMarker>(
    ir: &GraphIR<B>,
    a: AnnotatedNode,
    b: AnnotatedNode,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let bn = ir.get(b.idx)?;

    if bn.num_children == 1 {
        if let Some(Some(Concat { a: x, b: y })) = bn.parent_operation.as_ref().map(downcast) {
            let an = ir.get(a.idx)?;
            let xn = ir.get(x.idx)?;
            let yn = ir.get(y.idx)?;

            // exchange only worth it if the extraction of `a`
            // into pieces can be amortised by batching on `b`
            if !an.info.batched && xn.info.batched && yn.info.batched {
                let a_flat = Shape::new(a.shape.size(), 1);
                let a_resh = AnnotatedNode { idx: a.idx, shape: a_flat };

                let a_lower_shape = Shape::new(a.shape.rows(), x.shape.rows());
                let a_lower_data = GraphIRNode {
                    idx: ir.new_idx(),
                    info: NodeInfo { shape: Shape::new(a_lower_shape.size(), 1), ..an.info },
                    parent_operation: Some(Box::new(Slice { input: a_resh, start: 0, end: a_lower_shape.size() })),
                    id: None,
                    num_children: 0,
                };
                let a_lower = AnnotatedNode { idx: a_lower_data.idx, shape: a_lower_shape };

                let a_upper_shape = Shape::new(a.shape.rows(), y.shape.rows());
                let a_upper_data = GraphIRNode {
                    idx: ir.new_idx(),
                    info: NodeInfo { shape: Shape::new(a_upper_shape.size(), 1), ..an.info },
                    parent_operation: Some(Box::new(Slice {
                        input: a_resh,
                        start: a_lower_shape.size(),
                        end: a_flat.size(),
                    })),
                    id: None,
                    num_children: 0,
                };
                let a_upper = AnnotatedNode { idx: a_upper_data.idx, shape: a_upper_shape };

                let ab_lower_data = GraphIRNode {
                    idx: ir.new_idx(),
                    info: old_data.info,
                    parent_operation: Some(Box::new(Matmul { a: a_lower, b: *x, transa: false, transb: false })),
                    id: None,
                    num_children: 0,
                };
                let ab_lower = AnnotatedNode { idx: ab_lower_data.idx, shape: ab_lower_data.info.shape };

                let ab_upper_data = GraphIRNode {
                    idx: ir.new_idx(),
                    info: old_data.info,
                    parent_operation: Some(Box::new(Matmul { a: a_upper, b: *y, transa: false, transb: false })),
                    id: None,
                    num_children: 0,
                };
                let ab_upper = AnnotatedNode { idx: ab_upper_data.idx, shape: ab_upper_data.info.shape };

                let new_data =
                    old_data.with_new_op(LinearCombination { alpha: 1.0, beta: 1.0, a: ab_lower, b: ab_upper });

                return GraphIRTransform::new(
                    &[b.idx],
                    vec![a_lower_data, a_upper_data, ab_lower_data, ab_upper_data, new_data],
                );
            }
        }
    }

    Ok(None)
}

fn fuse_diffable_from_output<B: BackendMarker>(
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
                return GraphIRTransform::new(&[node.idx], vec![new_data]);
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
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&LinearCombination { a, b, alpha: 1.0, beta: -1.0 }) = downcast(op) {
                if a.idx != b.idx && ir.get(a.idx)?.info.batched == ir.get(b.idx)?.info.batched {
                    let new_data = old_data.with_new_op(AbsPowerError { a, b, power });
                    return GraphIRTransform::new(&[node.idx], [new_data]);
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
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(node.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&LinearCombination { a, b, alpha, beta }) = downcast(op) {
                let new_data =
                    old_data.with_new_op(LinearCombination { a, b, alpha: alpha * scale, beta: beta * scale });
                return GraphIRTransform::new(&[node.idx], [new_data]);
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
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
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
                return GraphIRTransform::new(&[lhs.idx], [new_data]);
            }

            if let Some(&Matmul { a, transa: false, b, transb: false }) = downcast(op) {
                if !ir.get(rhs.idx)?.info.batched {
                    let new_data = old_data.with_new_op(Affine { weights: a, inputs: b, biases: *rhs });
                    return GraphIRTransform::new(&[lhs.idx], [new_data]);
                }
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
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let ir_node = ir.get(lhs.idx)?;

    if ir_node.num_children == 1 {
        if let Some(op) = &ir_node.parent_operation {
            if let Some(&Unary { input, op: UnaryOp::Mul(x) }) = downcast(op) {
                let new_data = old_data.with_new_op(LinearCombination { a: input, b: *rhs, alpha: alpha * x, beta });
                return GraphIRTransform::new(&[lhs.idx], vec![new_data]);
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
