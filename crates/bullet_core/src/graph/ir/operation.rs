pub mod affine;
pub mod binary;
pub mod nary;
pub mod sparse;
pub mod unary;
pub mod util;

use std::{fmt, num::NonZeroUsize};

use acyclib::graph::NodeId;

use crate::{
    device::Device,
    function::DeviceFunction,
    graph::{
        Graph,
        ir::{BackendMarker, GraphIR, node::NodeInfo},
    },
};

use super::{GraphIRError, Shape, node::AnnotatedNode};

pub trait GraphIROperationBase<B: BackendMarker>: std::any::Any + std::fmt::Debug + 'static {
    fn nodes(&self) -> Vec<AnnotatedNode>;

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError>;

    fn output_batched(&self, ir: &GraphIR<B>) -> Result<bool, GraphIRError> {
        Ok(self.nodes().iter().any(|node| ir.get(node.idx).unwrap().ty().batched))
    }

    fn output_requires_grad(&self, ir: &GraphIR<B>) -> Result<bool, GraphIRError> {
        Ok(self.nodes().iter().any(|node| ir.get(node.idx).unwrap().ty().requires_grad))
    }

    fn output_layout(&self, _ir: &GraphIR<B>) -> Result<Option<NonZeroUsize>, GraphIRError> {
        Ok(None)
    }

    fn ancillary_buffers(&self, _ir: &GraphIR<B>) -> Result<Vec<(Shape, Option<NonZeroUsize>, bool)>, GraphIRError> {
        Ok(Vec::new())
    }

    fn shorthand(&self) -> String {
        let dbg = format!("{self:?}");
        dbg.split_whitespace().next().unwrap().to_string()
    }
}

pub trait GraphIROperationCompilable<B: BackendMarker>: GraphIROperationBase<B>
where
    B::Backend: Device,
{
    fn forward_pass(&self, ir: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend>;

    fn backward_pass(&self, ir: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend>;
}

#[derive(Clone)]
pub struct GraphIRLeaf {
    pub id: Option<String>,
    pub ty: NodeInfo,
}

impl fmt::Debug for GraphIRLeaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id.clone().unwrap_or("__unknown".to_string()))
    }
}

impl<B: BackendMarker> GraphIROperationBase<B> for GraphIRLeaf {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        Vec::new()
    }

    fn output_batched(&self, _ir: &GraphIR<B>) -> Result<bool, GraphIRError> {
        Ok(self.ty.batched)
    }

    fn output_layout(&self, _ir: &GraphIR<B>) -> Result<Option<NonZeroUsize>, GraphIRError> {
        Ok(self.ty.sparse)
    }

    fn output_requires_grad(&self, _ir: &GraphIR<B>) -> Result<bool, GraphIRError> {
        Ok(self.ty.requires_grad)
    }

    fn output_shape(&self, _ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        Ok(self.ty.shape)
    }

    fn shorthand(&self) -> String {
        self.id.clone().unwrap_or("__unknown".to_string())
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for GraphIRLeaf {
    fn forward_pass(&self, _: &Graph<B::Backend>, _: NodeId) -> DeviceFunction<B::Backend> {
        DeviceFunction::default()
    }

    fn backward_pass(&self, _: &Graph<B::Backend>, _: NodeId) -> DeviceFunction<B::Backend> {
        DeviceFunction::default()
    }
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
