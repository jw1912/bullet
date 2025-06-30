use crate::graph_ir::{node::AnnotatedNode, operation::GraphIROperationError, shape::Shape, GraphIR};

pub fn check_dense_eq(ir: &GraphIR, node: &AnnotatedNode, dense: bool) -> Result<(), GraphIROperationError> {
    if ir.get(node.idx).unwrap().sparse.is_none() == dense {
        Ok(())
    } else {
        Err(GraphIROperationError::IncorrectDataLayout)
    }
}

pub fn check_not_batched(ir: &GraphIR, node: &AnnotatedNode) -> Result<(), GraphIROperationError> {
    if ir.get(node.idx).unwrap().batched {
        Err(GraphIROperationError::BatchedInputNotSupported)
    } else {
        Ok(())
    }
}

pub fn check_matmul(a: Shape, b: Shape) -> Result<Shape, GraphIROperationError> {
    if let Some(c) = a.matmul(b) {
        Ok(c)
    } else {
        Err(GraphIROperationError::InvalidMatmulDims)
    }
}

pub fn check_same_batching(ir: &GraphIR, x: &[&AnnotatedNode]) -> Result<(), GraphIROperationError> {
    if x.iter().all(|y| ir.get(y.idx).unwrap().batched == ir.get(x[0].idx).unwrap().batched) {
        Ok(())
    } else {
        Err(GraphIROperationError::MismatchedBatching)
    }
}

pub fn check_no_grad(ir: &GraphIR, x: &[&AnnotatedNode]) -> Result<(), GraphIROperationError> {
    if x.iter().any(|y| ir.get(y.idx).unwrap().requires_grad) {
        Err(GraphIROperationError::GradientNotSupported)
    } else {
        Ok(())
    }
}
