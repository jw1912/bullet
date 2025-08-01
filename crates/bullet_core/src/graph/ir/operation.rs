pub mod affine;
pub mod binary;
pub mod sparse;
pub mod unary;
pub mod util;

use std::num::NonZeroUsize;

use crate::{
    device::Device,
    graph::{
        ir::{BackendMarker, GraphIRNodeInfo},
        GraphFunction,
    },
};

use super::{node::AnnotatedNode, GraphIR, GraphIRError, Shape};

pub trait GraphIROperation<B: BackendMarker>: std::any::Any + std::fmt::Debug + 'static {
    fn nodes(&self) -> Vec<AnnotatedNode>;

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError>;

    fn output_batched(&self, ir: &GraphIR<B>) -> Result<bool, GraphIRError> {
        Ok(self.nodes().iter().any(|node| ir.get(node.idx).unwrap().info.batched))
    }

    fn output_requires_grad(&self, ir: &GraphIR<B>) -> Result<bool, GraphIRError> {
        Ok(self.nodes().iter().any(|node| ir.get(node.idx).unwrap().info.requires_grad))
    }

    fn ancillary_buffers(&self, _ir: &GraphIR<B>) -> Result<Vec<(Shape, Option<NonZeroUsize>)>, GraphIRError> {
        Ok(Vec::new())
    }

    fn shorthand(&self) -> String {
        let dbg = format!("{self:?}");
        dbg.split_whitespace().next().unwrap().to_string()
    }
}

pub trait GraphIROperationCompilable<B: BackendMarker>: GraphIROperation<B>
where
    B::Backend: Device,
{
    fn forward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend>;

    fn backward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend>;
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
