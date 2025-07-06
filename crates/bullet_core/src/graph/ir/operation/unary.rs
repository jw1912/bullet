use crate::graph::ir::{
    node::AnnotatedNode,
    operation::{util, GraphIROperation, GraphIROperationError},
    shape::Shape,
    GraphIR, GraphIRError,
};

/// List of supported activation functions.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffableFromOutput {
    Identity = 0,
    ReLU = 1,
    CReLU = 2,
    SCReLU = 3,
    SqrReLU = 4,
    Sigmoid = 5,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Unary {
    pub input: AnnotatedNode,
    pub op: UnaryOp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnaryOp {
    DiffableFromOutput(DiffableFromOutput),
    Add(f32),
    Mul(f32),
    AbsPow(f32),
}

impl GraphIROperation for Unary {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;

        Ok(self.input.shape)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reduce {
    Sum,
    Avg,
}

#[derive(Debug)]
pub struct ReduceAcrossBatch {
    pub input: AnnotatedNode,
    pub reduction: Reduce,
}

impl GraphIROperation for ReduceAcrossBatch {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_batched(&self, _: &GraphIR) -> Result<bool, GraphIRError> {
        Ok(false)
    }

    fn output_shape(&self, ir: &GraphIR) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;

        Ok(self.input.shape)
    }
}

#[derive(Debug)]
pub struct PairwiseMul {
    pub input: AnnotatedNode,
    pub post_concat: bool,
}

impl GraphIROperation for PairwiseMul {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;

        let is = self.input.shape;
        let min = 2 + 2 * usize::from(self.post_concat);

        if is.rows() % min == 0 {
            Ok(Shape::new(is.rows() / 2, is.cols()))
        } else {
            Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(is)))
        }
    }
}

#[derive(Debug)]
pub struct Slice {
    pub input: AnnotatedNode,
    pub start: usize,
    pub end: usize,
}

impl GraphIROperation for Slice {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        let is = self.input.shape;
        if self.end > self.start && self.end <= is.rows() && is.cols() == 1 {
            Ok(Shape::new(self.end - self.start, 1))
        } else {
            Err(GraphIRError::Op(GraphIROperationError::OutOfBounds(is, [self.start, self.end])))
        }
    }
}

#[derive(Debug)]
pub struct ToDense(pub AnnotatedNode);

impl GraphIROperation for ToDense {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.0]
    }

    fn output_shape(&self, ir: &GraphIR) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.0, false)?;
        Ok(self.0.shape)
    }
}

#[derive(Debug)]
pub struct Copy {
    pub input: AnnotatedNode,
    pub stop_grad: bool,
}

impl GraphIROperation for Copy {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;

        Ok(self.input.shape)
    }

    fn output_requires_grad(&self, ir: &GraphIR) -> Result<bool, GraphIRError> {
        Ok(!self.stop_grad && ir.get(self.input.idx).unwrap().requires_grad)
    }
}
