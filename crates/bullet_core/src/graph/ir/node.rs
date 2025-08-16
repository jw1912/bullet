use std::num::NonZeroUsize;

use acyclib::graph::NodeId;

use super::shape::Shape;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeInfo {
    pub requires_grad: bool,
    pub sparse: Option<NonZeroUsize>,
    pub batched: bool,
    pub shape: Shape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnnotatedNode {
    pub idx: NodeId,
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
