use std::collections::HashMap;

use crate::graph::{
    instruction,
    ir::{
        node::AnnotatedNode,
        operation::{unary::Reduce, util, GraphIROperation, GraphIROperationCompilable, GraphIROperationError},
        shape::Shape,
        BackendMarker, GraphIR, GraphIRError, GraphIRNodeInfo,
    },
    GraphFunction, NodeId, NodeIdTy,
};

#[derive(Clone, Debug, PartialEq)]
pub struct LinearCombination {
    pub items: Vec<(usize, f32)>,
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

impl<B: BackendMarker> GraphIROperation<B> for LinearCombination {
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
    fn forward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let output = NodeId::new(output_node, NodeIdTy::Values);
        let bsn = util::batch_size_node(node_info, &<Self as GraphIROperation<B>>::nodes(self));

        let mut func = GraphFunction::default();

        func.push(instruction::MaybeUpdateBatchSize { input: NodeId::new(bsn, NodeIdTy::Values), output });

        let mut push = |input_mul, output_mul, node| {
            if !node_info.get(output_node).unwrap().batched || node_info.get(node).unwrap().batched {
                func.push(instruction::LinearCombination {
                    input_mul,
                    output_mul,
                    input: NodeId::new(node, NodeIdTy::Values),
                    output,
                });
            } else {
                func.push(instruction::LinearCombinationSplat {
                    input_mul,
                    output_mul,
                    input: NodeId::new(node, NodeIdTy::Values),
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

    fn backward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let input = NodeId::new(output_node, NodeIdTy::Gradients);

        let mut func = GraphFunction::default();

        let mut push = |input_mul, output_mul, node| {
            if node_info.get(node).unwrap().requires_grad {
                let output = NodeId::new(node, NodeIdTy::Gradients);

                func.push(instruction::MaybeUpdateBatchSize { input: NodeId::new(node, NodeIdTy::Values), output });

                if !node_info.get(output_node).unwrap().batched || node_info.get(node).unwrap().batched {
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
