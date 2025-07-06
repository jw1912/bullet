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
pub struct Binary {
    pub a: AnnotatedNode,
    pub b: AnnotatedNode,
    pub op: BinaryOp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinaryOp {
    LinearCombination { alpha: f32, beta: f32 },
    PowerError { power: f32 },
    SoftmaxCrossEntropyLoss,
}

impl GraphIROperation for Binary {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.a, self.b]
    }

    fn output_shape(&self, ir: &GraphIR) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.a, true)?;
        util::check_dense_eq(ir, &self.b, true)?;

        if self.a.shape == self.b.shape {
            Ok(self.a.shape)
        } else {
            Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![self.a.shape, self.b.shape])))
        }
    }
}

#[derive(Debug)]
pub struct Concat {
    pub a: AnnotatedNode,
    pub b: AnnotatedNode,
}

impl GraphIROperation for Concat {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.a, self.b]
    }

    fn output_shape(&self, ir: &GraphIR) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.a, true)?;
        util::check_dense_eq(ir, &self.b, true)?;
        util::check_same_batching(ir, &[&self.a, &self.b])?;

        let ash = self.a.shape;

        if ash.cols() != 1 {
            return Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(ash)));
        }

        if ash.cols() == self.b.shape.cols() {
            Ok(Shape::new(ash.rows() + self.b.shape.rows(), ash.cols()))
        } else {
            Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![ash, self.b.shape])))
        }
    }
}

#[derive(Debug)]
pub struct Select {
    pub input: AnnotatedNode,
    pub buckets: AnnotatedNode,
}

impl GraphIROperation for Select {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input, self.buckets]
    }

    fn output_shape(&self, ir: &GraphIR) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        util::check_dense_eq(ir, &self.buckets, false)?;
        util::check_same_batching(ir, &[&self.input, &self.buckets])?;
        let is = self.input.shape;
        let bs = self.buckets.shape;

        if is.cols() == bs.cols() && is.rows() % bs.rows() == 0 {
            Ok(Shape::new(is.rows() / bs.rows(), is.cols()))
        } else {
            Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![is, bs])))
        }
    }
}
