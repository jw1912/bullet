use std::num::NonZeroUsize;

use super::{op::GraphIROp, Shape};

#[derive(Clone, Debug)]
pub struct GraphIRNode {
    pub idx: usize,
    pub shape: Shape,
    pub sparse: Option<NonZeroUsize>,
    pub batched: bool,
    pub requires_grad: bool,
    pub parent_operation: Option<GraphIROp>,
    pub num_children: usize,
    pub id: Option<String>,
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
