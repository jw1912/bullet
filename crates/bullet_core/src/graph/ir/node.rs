use std::num::NonZeroUsize;

use super::{operation::GraphIROperation, shape::Shape};

#[derive(Clone, Copy, Debug)]
pub struct NodeInfo {
    pub requires_grad: bool,
    pub sparse: Option<NonZeroUsize>,
    pub batched: bool,
    pub shape: Shape,
}

#[derive(Debug)]
pub struct GraphIRNode {
    pub idx: usize,
    pub info: NodeInfo,
    pub parent_operation: Option<Box<dyn GraphIROperation>>,
    pub num_children: usize,
    pub id: Option<String>,
}

impl GraphIRNode {
    pub fn with_new_op(&self, op: impl GraphIROperation) -> Self {
        Self {
            idx: self.idx,
            info: self.info,
            parent_operation: Some(Box::new(op)),
            num_children: self.num_children,
            id: self.id.clone(),
        }
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
