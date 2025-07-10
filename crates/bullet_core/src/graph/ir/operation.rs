pub mod affine;
pub mod binary;
pub mod sparse;
pub mod unary;
pub mod util;

use crate::{backend::device::Device, graph::GraphFunction};

use super::{node::AnnotatedNode, GraphIR, GraphIRError, Shape};

pub trait GraphIROperation: std::any::Any + std::fmt::Debug + 'static {
    fn nodes(&self) -> Vec<AnnotatedNode>;

    fn output_shape(&self, ir: &GraphIR) -> Result<Shape, GraphIRError>;

    fn output_batched(&self, ir: &GraphIR) -> Result<bool, GraphIRError> {
        Ok(self.nodes().iter().any(|node| ir.get(node.idx).unwrap().batched))
    }

    fn output_requires_grad(&self, ir: &GraphIR) -> Result<bool, GraphIRError> {
        Ok(self.nodes().iter().any(|node| ir.get(node.idx).unwrap().requires_grad))
    }
}

pub trait GraphIROperationCompile<D: Device>: GraphIROperation {
    fn forward_pass(&self) -> GraphFunction<D>;

    fn backward_pass(&self) -> GraphFunction<D>;
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphIROperationError {
    InvalidInputShape(Shape),
    MismatchedInputShapes(Vec<Shape>),
    OutOfBounds(Shape, [usize; 2]),
    IncorrectDataLayout,
    BatchedInputNotSupported,
    InvalidMatmulDims,
    AnnotatedNodeWithIdAlreadyExists,
    MismatchedBatching,
    GradientNotSupported,
}
