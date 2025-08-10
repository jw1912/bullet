use crate::graph::{
    builder::Shape,
    ir::{
        node::AnnotatedNode,
        operation::{
            affine::Matmul,
            binary::{Concat, LinearCombination, Select},
            unary::{Slice, Unary, UnaryOp},
        },
        transform::GraphIRTransform,
        BackendMarker, GraphIR, GraphIRError, GraphIRNode,
    },
};

use super::downcast;

pub fn select<B: BackendMarker>(
    ir: &GraphIR<B>,
    input: AnnotatedNode,
    buckets: AnnotatedNode,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let node = ir.get(input.idx)?;

    if node.num_children == 1 {
        if let Some(Some(&LinearCombination { alpha, beta, a, b })) = node.parent_operation.as_ref().map(downcast) {
            let lhs_data = ir.make_result_of_op(Select { input: a, buckets })?;
            let lhs = AnnotatedNode { idx: lhs_data.idx, shape: lhs_data.info.shape };

            let rhs_data = ir.make_result_of_op(Select { input: b, buckets })?;
            let rhs = AnnotatedNode { idx: rhs_data.idx, shape: rhs_data.info.shape };

            let new_data = old_data.with_new_op(LinearCombination { alpha, beta, a: lhs, b: rhs });

            return GraphIRTransform::new(&[node.idx], vec![lhs_data, rhs_data, new_data]);
        }

        if let Some(Some(&Unary { input, op })) = node.parent_operation.as_ref().map(downcast) {
            let select_data = ir.make_result_of_op(Select { input, buckets })?;
            let select = AnnotatedNode { idx: select_data.idx, shape: select_data.info.shape };

            let new_data = old_data.with_new_op(Unary { input: select, op });

            return GraphIRTransform::new(&[node.idx], vec![select_data, new_data]);
        }
    }

    Ok(None)
}

pub fn unary_concat<B: BackendMarker>(
    ir: &GraphIR<B>,
    input: AnnotatedNode,
    op: UnaryOp,
    old_data: &GraphIRNode<B>,
) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
    let node = ir.get(input.idx)?;

    if node.num_children == 1 {
        if let Some(Some(Concat { a, b })) = node.parent_operation.as_ref().map(downcast) {
            let lower_data = ir.make_result_of_op(Unary { input: *a, op })?;
            let lower = AnnotatedNode { idx: lower_data.idx, shape: a.shape };

            let upper_data = ir.make_result_of_op(Unary { input: *b, op })?;
            let upper = AnnotatedNode { idx: upper_data.idx, shape: b.shape };

            let new_data = old_data.with_new_op(Concat { a: lower, b: upper });

            return GraphIRTransform::new(&[node.idx], vec![lower_data, upper_data, new_data]);
        }
    }

    Ok(None)
}

pub fn matmul_concat<B: BackendMarker>(
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
                let a_lower_data =
                    ir.make_result_of_op(Slice { input: a_resh, start: 0, end: a_lower_shape.size() })?;
                let a_lower = AnnotatedNode { idx: a_lower_data.idx, shape: a_lower_shape };

                let a_upper_shape = Shape::new(a.shape.rows(), y.shape.rows());
                let a_upper_data =
                    ir.make_result_of_op(Slice { input: a_resh, start: a_lower_shape.size(), end: a_flat.size() })?;
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
