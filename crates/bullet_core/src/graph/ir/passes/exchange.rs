use acyclib::graph::NodeId;

use crate::graph::{
    builder::Shape,
    ir::{
        BackendMarker, GraphIR, GraphIRError, GraphIRMethods,
        node::AnnotatedNode,
        operation::{
            GraphIROperationCompilable,
            affine::Matmul,
            binary::{Concat, Select},
            nary::LinearCombination,
            unary::{Slice, Unary},
        },
        passes::GraphIRSimplePass,
    },
};

use super::downcast;

#[derive(Debug)]
pub struct ExchangeElementwiseAndSelect;

impl<B: BackendMarker> GraphIRSimplePass<B> for ExchangeElementwiseAndSelect
where
    Select: GraphIROperationCompilable<B>,
{
    fn try_pass_on_node(&self, ir: &mut GraphIR<B>, target: NodeId) -> Result<bool, GraphIRError> {
        let old_data = ir.get(target)?;

        if let Some(Select { input, buckets }) = downcast(old_data.op()) {
            let parent = ir.get(input.idx)?;

            if parent.children() == 1 {
                if let Some(LinearCombination { items, shape }) = downcast(parent.op()) {
                    let mut new = Vec::new();

                    for (node, weight) in items {
                        let input = AnnotatedNode { idx: node, shape };
                        new.push((ir.create(Select { input, buckets })?, weight));
                    }

                    ir.replace(target, LinearCombination::new(new)?)?;

                    return Ok(true);
                }
            }

            if parent.children() == 1 {
                if let Some(Unary { input, op }) = downcast(parent.op()) {
                    let input = ir.create(Select { input, buckets })?;
                    ir.replace(target, Unary { input, op })?;

                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

#[derive(Debug)]
pub struct ExchangeConcatAndUnary;

impl<B: BackendMarker> GraphIRSimplePass<B> for ExchangeConcatAndUnary {
    fn try_pass_on_node(&self, ir: &mut GraphIR<B>, target: NodeId) -> Result<bool, GraphIRError> {
        let old_data = ir.get(target)?;

        if let Some(Unary { input, op }) = downcast(old_data.op()) {
            let parent = ir.get(input.idx)?;

            if parent.children() == 1 {
                if let Some(Concat { a, b }) = downcast(parent.op()) {
                    let lower = ir.create(Unary { input: a, op })?;
                    let upper = ir.create(Unary { input: b, op })?;

                    ir.replace(target, Concat { a: lower, b: upper })?;

                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

#[derive(Debug)]
pub struct ExchangeMatmulAndConcatWithSliceAndMatmul;

impl<B: BackendMarker> GraphIRSimplePass<B> for ExchangeMatmulAndConcatWithSliceAndMatmul {
    fn try_pass_on_node(&self, ir: &mut GraphIR<B>, target: NodeId) -> Result<bool, GraphIRError> {
        let old_data = ir.get(target)?;

        if let Some(Matmul { a, b, transa: false, transb: false }) = downcast(old_data.op()) {
            let bn = ir.get(b.idx)?;

            if bn.children() == 1 {
                if let Some(Concat { a: x, b: y }) = downcast(bn.op()) {
                    let an = ir.get(a.idx)?.ty();
                    let xn = ir.get(x.idx)?.ty();
                    let yn = ir.get(y.idx)?.ty();

                    // exchange only worth it if the extraction of `a`
                    // into pieces can be amortised by batching on `b`
                    if !an.batched && xn.batched && yn.batched {
                        let flat = Shape::new(a.shape.size(), 1);
                        let resh = AnnotatedNode { idx: a.idx, shape: flat };

                        let lower_shape = Shape::new(a.shape.rows(), x.shape.rows());
                        let lower = ir.create(Slice { input: resh, start: 0, end: lower_shape.size() })?;
                        let lower = AnnotatedNode { idx: lower.idx, shape: lower_shape };

                        let upper_shape = Shape::new(a.shape.rows(), y.shape.rows());
                        let upper = ir.create(Slice { input: resh, start: lower_shape.size(), end: flat.size() })?;
                        let upper = AnnotatedNode { idx: upper.idx, shape: upper_shape };

                        let ab_lower = ir.create(Matmul { a: lower, b: x, transa: false, transb: false })?;

                        let ab_upper = ir.create(Matmul { a: upper, b: y, transa: false, transb: false })?;

                        ir.replace(target, LinearCombination::new([(ab_lower, 1.0), (ab_upper, 1.0)])?)?;

                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }
}
