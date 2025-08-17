use acyclib::graph::NodeId;

use crate::graph::{
    Graph, GraphNodeIdTy,
    ir::{BackendMarker, GraphIR, node::AnnotatedNode, operation::GraphIROperationError, shape::Shape},
};

pub fn check_dense_eq<B: BackendMarker>(
    ir: &GraphIR<B>,
    node: &AnnotatedNode,
    dense: bool,
) -> Result<(), GraphIROperationError> {
    if ir.get(node.idx).unwrap().ty().sparse.is_none() == dense {
        Ok(())
    } else {
        Err(GraphIROperationError::IncorrectDataLayout)
    }
}

pub fn check_not_batched<B: BackendMarker>(ir: &GraphIR<B>, node: &AnnotatedNode) -> Result<(), GraphIROperationError> {
    if ir.get(node.idx).unwrap().ty().batched {
        Err(GraphIROperationError::BatchedInputNotSupported)
    } else {
        Ok(())
    }
}

pub fn check_matmul(a: Shape, b: Shape) -> Result<Shape, GraphIROperationError> {
    if let Some(c) = a.matmul(b) { Ok(c) } else { Err(GraphIROperationError::InvalidMatmulDims) }
}

pub fn check_same_batching<B: BackendMarker>(
    ir: &GraphIR<B>,
    x: &[&AnnotatedNode],
) -> Result<(), GraphIROperationError> {
    if x.iter().all(|y| ir.get(y.idx).unwrap().ty().batched == ir.get(x[0].idx).unwrap().ty().batched) {
        Ok(())
    } else {
        Err(GraphIROperationError::MismatchedBatching)
    }
}

pub fn check_no_grad<B: BackendMarker>(ir: &GraphIR<B>, x: &[&AnnotatedNode]) -> Result<(), GraphIROperationError> {
    if x.iter().any(|y| ir.get(y.idx).unwrap().ty().requires_grad) {
        Err(GraphIROperationError::GradientNotSupported)
    } else {
        Ok(())
    }
}

pub fn batch_size_node<B: BackendMarker>(graph: &Graph<B::Backend>, nodes: &[AnnotatedNode]) -> NodeId {
    nodes
        .iter()
        .find(|x| graph.get_ref(x.idx, GraphNodeIdTy::Values).borrow().batch_size().is_some())
        .unwrap_or(&nodes[0])
        .idx
}
