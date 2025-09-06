use std::{fmt, num::NonZeroUsize};

use crate::{dag::NodeId, device::tensor::Shape};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct NodeInfo {
    pub requires_grad: bool,
    pub sparse: Option<NonZeroUsize>,
    pub batched: bool,
    pub shape: Shape,
}

impl fmt::Debug for NodeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.requires_grad {
            write!(f, "gr_")?;
        }

        if let Some(val) = self.sparse {
            write!(f, "sp{}_", val.get())?;
        }

        if self.batched {
            write!(f, "?x")?;
        }

        write!(f, "{:?}", self.shape)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AnnotatedNode {
    pub idx: NodeId,
    pub shape: Shape,
}

impl fmt::Debug for AnnotatedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.shape)
    }
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
