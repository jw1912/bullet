use std::num::NonZeroUsize;

use super::{op::GraphIROp, GraphIR, GraphIRError, Shape};

#[derive(Clone, Debug)]
pub struct GraphIRNode {
    pub id: Option<String>,
    pub size: usize,
    pub requires_grad: bool,
    pub parent_operation: Option<GraphIROp>,
    pub num_children: usize,
    pub own: AnnotatedNode,
    pub sparse: Option<NonZeroUsize>,
    pub can_be_batched: bool,
}

impl GraphIRNode {
    pub fn is_valid(&self, ir: &GraphIR) -> Result<(), GraphIRError> {
        if self.own.shape.size() != self.size {
            return Err(GraphIRError::Node(GraphIRNodeError::NodeDataDoesNotMatchExpected));
        }

        if let Some(op) = &self.parent_operation {
            let (shape, batched) = op.output_info(ir)?;

            if shape != self.own.shape || self.can_be_batched != batched {
                return Err(GraphIRError::Node(GraphIRNodeError::NodeDataDoesNotMatchExpected));
            }
        }

        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub enum GraphIRNodeError {
    NodeWithIdAlreadyExists(String),
    NodeDataDoesNotMatchExpected,
    NodeDoesNotExist,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnnotatedNode {
    pub idx: usize,
    pub shape: Shape,
}

#[derive(Debug)]
pub enum AnnotatedNodeError {
    ShapeSizesDiffer(Shape, Shape),
    NotSparse,
}

impl AnnotatedNode {
    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn reshape(mut self, shape: Shape) -> Result<Self, AnnotatedNodeError> {
        (self.shape.size() == shape.size())
            .then(|| {
                self.shape = shape;
                self
            })
            .ok_or(AnnotatedNodeError::ShapeSizesDiffer(self.shape, shape))
    }
}
