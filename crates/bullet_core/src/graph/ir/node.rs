use std::num::{NonZero, NonZeroUsize};

use crate::backend::device::blas::Shape;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnnotatedNode {
    pub idx: usize,
    pub shape: Shape,
    pub sparse: Option<NonZeroUsize>,
    pub can_be_batched: bool,
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

    pub fn is_sparse(&self) -> bool {
        self.sparse.is_some()
    }

    pub fn nnz(&self) -> Result<usize, AnnotatedNodeError> {
        self.sparse.map(NonZero::get).ok_or(AnnotatedNodeError::NotSparse)
    }
}
