use std::collections::HashMap;

use acyclib::graph::NodeId;

use crate::graph::{
    GraphFunction, GraphNodeId, GraphNodeIdTy, instruction,
    ir::{
        BackendMarker, GraphIR, GraphIRError,
        node::AnnotatedNode,
        operation::{GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError, unary::Reduce, util},
        shape::Shape,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub struct LinearCombination {
    pub items: Vec<(NodeId, f32)>,
    pub shape: Shape,
}

impl LinearCombination {
    pub fn new(items: impl Into<Vec<(AnnotatedNode, f32)>>) -> Result<Self, GraphIRError> {
        let mut unique = HashMap::new();

        let items: Vec<(AnnotatedNode, f32)> = items.into();

        let shape = items[0].0.shape;

        for (node, weight) in items {
            if node.shape != shape {
                return Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![shape, node.shape])));
            }

            if let Some(w) = unique.insert(node.idx, weight) {
                *unique.get_mut(&node.idx).unwrap() += w;
            }
        }

        Ok(Self { items: unique.into_iter().collect(), shape })
    }
}

impl<B: BackendMarker> GraphIROperationBase<B> for LinearCombination {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        let shape = self.shape;
        self.items.iter().map(|x| AnnotatedNode { idx: x.0, shape }).collect()
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        let shape = self.shape;

        for &(idx, _) in &self.items {
            util::check_dense_eq(ir, &AnnotatedNode { idx, shape }, true)?;
        }

        Ok(shape)
    }

    fn shorthand(&self) -> String {
        match &self.items[..] {
            [(_, 1.0), (_, 1.0)] => "Add".to_string(),
            [(_, 1.0), (_, -1.0)] | [(_, -1.0), (_, 1.0)] => "Sub".to_string(),
            _ => format!("LinearCombination{:?}", self.items.iter().map(|x| x.1).collect::<Vec<_>>()),
        }
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for LinearCombination {
    fn forward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let output = GraphNodeId::new(output_node, GraphNodeIdTy::Values);
        let bsn = util::batch_size_node(ir, &<Self as GraphIROperationBase<B>>::nodes(self));

        let mut func = GraphFunction::default();

        func.push(instruction::MaybeUpdateBatchSize { input: GraphNodeId::new(bsn, GraphNodeIdTy::Values), output });

        let mut push = |input_mul, output_mul, node| {
            if !ir.get(output_node).unwrap().ty().batched || ir.get(node).unwrap().ty().batched {
                func.push(instruction::LinearCombination {
                    input_mul,
                    output_mul,
                    input: GraphNodeId::new(node, GraphNodeIdTy::Values),
                    output,
                });
            } else {
                func.push(instruction::LinearCombinationSplat {
                    input_mul,
                    output_mul,
                    input: GraphNodeId::new(node, GraphNodeIdTy::Values),
                    output,
                });
            }
        };

        let mut base = 0.0;
        for &(node, weight) in &self.items {
            push(weight, base, node);
            base = 1.0;
        }

        func
    }

    fn backward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let input = GraphNodeId::new(output_node, GraphNodeIdTy::Gradients);

        let mut func = GraphFunction::default();

        let mut push = |input_mul, output_mul, node| {
            if ir.get(node).unwrap().ty().requires_grad {
                let output = GraphNodeId::new(node, GraphNodeIdTy::Gradients);

                func.push(instruction::MaybeUpdateBatchSize {
                    input: GraphNodeId::new(node, GraphNodeIdTy::Values),
                    output,
                });

                if !ir.get(output_node).unwrap().ty().batched || ir.get(node).unwrap().ty().batched {
                    func.push(instruction::LinearCombination { input_mul, output_mul, input, output });
                } else {
                    func.push(instruction::ReduceAcrossBatch {
                        input_mul,
                        output_mul,
                        input,
                        output,
                        reduction: Reduce::Sum,
                    });
                }
            }
        };

        for &(node, weight) in &self.items {
            push(weight, 1.0, node);
        }

        func
    }
}
