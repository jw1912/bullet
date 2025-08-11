use crate::graph::{
    builder::Shape,
    ir::{
        node::AnnotatedNode,
        operation::{
            affine::Matmul,
            binary::{Concat, Select},
            nary::LinearCombination,
            unary::{Slice, Unary},
        },
        passes::GraphIRSimplePass,
        transform::GraphIRTransform,
        BackendMarker, GraphIR, GraphIRError, GraphIRNode,
    },
};

use super::downcast;

pub struct ExchangeElementwiseAndSelect;

impl<B: BackendMarker> GraphIRSimplePass<B> for ExchangeElementwiseAndSelect {
    fn try_pass_on_node(&self, ir: &GraphIR<B>, node: usize) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
        let old_data = ir.get(node)?;

        if let Some(&Select { input, buckets }) = downcast(&old_data.parent_operation) {
            let node = ir.get(input.idx)?;

            if node.num_children == 1 {
                if let Some(LinearCombination { items, shape }) = downcast(&node.parent_operation) {
                    let shape = *shape;
                    let mut items: Vec<_> =
                        items.iter().copied().map(|(idx, weight)| (AnnotatedNode { idx, shape }, weight)).collect();
                    let mut new = Vec::new();

                    for (node, _) in &mut items {
                        let new_data = ir.result_of(Select { input: *node, buckets })?;
                        node.idx = new_data.idx;
                        node.shape = new_data.info.shape;
                        new.push(new_data);
                    }

                    new.push(old_data.with_new_op(LinearCombination::new(items)?));

                    return GraphIRTransform::new([node.idx], new);
                }

                if let Some(&Unary { input, op }) = downcast(&node.parent_operation) {
                    let select_data = ir.result_of(Select { input, buckets })?;
                    let select = AnnotatedNode { idx: select_data.idx, shape: select_data.info.shape };

                    let new_data = old_data.with_new_op(Unary { input: select, op });

                    return GraphIRTransform::new([node.idx], vec![select_data, new_data]);
                }
            }
        }

        Ok(None)
    }
}

pub struct ExchangeConcatAndUnary;

impl<B: BackendMarker> GraphIRSimplePass<B> for ExchangeConcatAndUnary {
    fn try_pass_on_node(&self, ir: &GraphIR<B>, node: usize) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
        let old_data = ir.get(node)?;

        if let Some(&Unary { input, op }) = downcast(&old_data.parent_operation) {
            let node = ir.get(input.idx)?;

            if node.num_children == 1 {
                if let Some(Concat { a, b }) = downcast(&node.parent_operation) {
                    let lower_data = ir.result_of(Unary { input: *a, op })?;
                    let lower = AnnotatedNode { idx: lower_data.idx, shape: a.shape };

                    let upper_data = ir.result_of(Unary { input: *b, op })?;
                    let upper = AnnotatedNode { idx: upper_data.idx, shape: b.shape };

                    let new_data = old_data.with_new_op(Concat { a: lower, b: upper });

                    return GraphIRTransform::new([node.idx], vec![lower_data, upper_data, new_data]);
                }
            }
        }

        Ok(None)
    }
}

pub struct ExchangeMatmulAndConcatWithSliceAndMatmul;

impl<B: BackendMarker> GraphIRSimplePass<B> for ExchangeMatmulAndConcatWithSliceAndMatmul {
    fn try_pass_on_node(&self, ir: &GraphIR<B>, node: usize) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
        let old_data = ir.get(node)?;

        if let Some(&Matmul { a, b, transa: false, transb: false }) = downcast(&old_data.parent_operation) {
            let bn = ir.get(b.idx)?;

            if bn.num_children == 1 {
                if let Some(Concat { a: x, b: y }) = downcast(&bn.parent_operation) {
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
                            ir.result_of(Slice { input: a_resh, start: 0, end: a_lower_shape.size() })?;
                        let a_lower = AnnotatedNode { idx: a_lower_data.idx, shape: a_lower_shape };

                        let a_upper_shape = Shape::new(a.shape.rows(), y.shape.rows());
                        let a_upper_data =
                            ir.result_of(Slice { input: a_resh, start: a_lower_shape.size(), end: a_flat.size() })?;
                        let a_upper = AnnotatedNode { idx: a_upper_data.idx, shape: a_upper_shape };

                        let ab_lower_data = GraphIRNode {
                            idx: ir.new_idx(),
                            info: old_data.info,
                            parent_operation: Some(Box::new(Matmul {
                                a: a_lower,
                                b: *x,
                                transa: false,
                                transb: false,
                            })),
                            id: None,
                            num_children: 0,
                        };
                        let ab_lower = AnnotatedNode { idx: ab_lower_data.idx, shape: ab_lower_data.info.shape };

                        let ab_upper_data = GraphIRNode {
                            idx: ir.new_idx(),
                            info: old_data.info,
                            parent_operation: Some(Box::new(Matmul {
                                a: a_upper,
                                b: *y,
                                transa: false,
                                transb: false,
                            })),
                            id: None,
                            num_children: 0,
                        };
                        let ab_upper = AnnotatedNode { idx: ab_upper_data.idx, shape: ab_upper_data.info.shape };

                        let new_data =
                            old_data.with_new_op(LinearCombination::new([(ab_lower, 1.0), (ab_upper, 1.0)])?);

                        return GraphIRTransform::new(
                            [b.idx],
                            vec![a_lower_data, a_upper_data, ab_lower_data, ab_upper_data, new_data],
                        );
                    }
                }
            }
        }

        Ok(None)
    }
}
