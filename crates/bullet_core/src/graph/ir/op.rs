use crate::backend::device::{base::Activation, blas::Shape};

use super::node::AnnotatedNode;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GraphIROp {
    Activate(AnnotatedNode, Activation),
    Affine(AnnotatedNode, AnnotatedNode, AnnotatedNode),
    SparseAffine(AnnotatedNode, AnnotatedNode, Option<AnnotatedNode>),
    SparseAffineDualActivate(AnnotatedNode, AnnotatedNode, AnnotatedNode, AnnotatedNode, Activation),
    Concat(AnnotatedNode, AnnotatedNode),
    Gather(AnnotatedNode, AnnotatedNode),
    LinearCombination(f32, AnnotatedNode, f32, AnnotatedNode),
    Mask(AnnotatedNode, AnnotatedNode),
    Matmul(AnnotatedNode, bool, AnnotatedNode, bool),
    PairwiseMul(AnnotatedNode, bool),
    PowerError(AnnotatedNode, AnnotatedNode, f32),
    ReduceAcrossBatch(AnnotatedNode),
    Select(AnnotatedNode, AnnotatedNode),
    Slice(AnnotatedNode, usize, usize),
    ToDense(AnnotatedNode),
    MaskedSoftmaxCrossEntropyLoss(AnnotatedNode, AnnotatedNode, AnnotatedNode),
    SoftmaxCrossEntropyLoss(AnnotatedNode, AnnotatedNode),
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphIROpError {
    pub op: Box<GraphIROp>,
    pub ty: GraphIROpErrorType,
}

impl GraphIROpError {
    pub fn new(op: &GraphIROp, ty: GraphIROpErrorType) -> Self {
        Self { op: Box::new(*op), ty }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphIROpErrorType {
    InvalidInputShape(Shape),
    MismatchedInputShapes(Vec<Shape>),
    OutOfBounds(Shape, [usize; 2]),
    IncorrectDataLayout,
    BatchedInputNotSupported,
    InvalidMatmulDims,
    ActivationCannotBeFused,
    AnnotatedNodeWithIdAlreadyExists,
}

impl GraphIROp {
    pub fn output_shape(&self) -> Result<Shape, GraphIROpError> {
        use GraphIROp::*;
        use GraphIROpErrorType::*;

        let ret = |cond, ok, err| if cond { Ok(ok) } else { Err(err) };

        let mismatch = |nodes: &[&AnnotatedNode]| GraphIROpError {
            op: Box::new(*self),
            ty: MismatchedInputShapes(nodes.iter().map(|&x| x.shape).collect::<Vec<_>>()),
        };

        let check_dense_eq = |node: &AnnotatedNode, dense: bool| {
            if node.sparse.is_none() == dense {
                Ok(())
            } else {
                Err(GraphIROpError::new(self, GraphIROpErrorType::IncorrectDataLayout))
            }
        };

        let check_not_batched = |node: &AnnotatedNode| {
            if node.can_be_batched {
                Err(GraphIROpError::new(self, GraphIROpErrorType::BatchedInputNotSupported))
            } else {
                Ok(())
            }
        };

        let check_matmul = |a: Shape, b: Shape| {
            if let Some(c) = a.matmul(b) {
                Ok(c)
            } else {
                Err(GraphIROpError::new(self, GraphIROpErrorType::InvalidMatmulDims))
            }
        };

        match self {
            Activate(node, _) => {
                check_dense_eq(node, true)?;
                Ok(node.shape)
            }
            Affine(w, i, b) => {
                check_dense_eq(w, true)?;
                check_dense_eq(i, true)?;
                check_dense_eq(b, true)?;
                check_not_batched(w)?;
                check_not_batched(b)?;

                let out = check_matmul(w.shape, i.shape)?;
                ret(out == b.shape, out, mismatch(&[w, i]))
            }
            Concat(a, b) => {
                check_dense_eq(a, true)?;
                check_dense_eq(b, true)?;

                if a.shape.cols() != 1 {
                    return Err(GraphIROpError::new(self, InvalidInputShape(a.shape)));
                }

                let out = Shape::new(a.shape.rows() + b.shape.rows(), a.shape.cols());
                ret(a.shape.cols() == b.shape.cols(), out, mismatch(&[a, b]))
            }
            Gather(input, mask) => {
                check_dense_eq(input, true)?;
                check_dense_eq(mask, false)?;

                let valid = input.shape.cols() == 1 && mask.shape.cols() == 1;
                ret(valid, mask.shape, mismatch(&[input, mask]))
            }
            LinearCombination(_, a, _, b) => {
                check_dense_eq(a, true)?;
                check_dense_eq(b, true)?;

                ret(a.shape == b.shape, a.shape, mismatch(&[a, b]))
            }
            Mask(input, mask) => {
                check_dense_eq(input, true)?;
                check_dense_eq(mask, false)?;

                ret(input.shape == mask.shape, input.shape, mismatch(&[input, mask]))
            }
            Matmul(a, transa, b, transb) => {
                check_dense_eq(a, true)?;
                check_dense_eq(b, true)?;

                let out = check_matmul(a.shape.maybe_transpose(*transa), b.shape.maybe_transpose(*transb))?;
                ret(true, out, mismatch(&[a, b]))
            }
            PairwiseMul(input, post_concat) => {
                let is = input.shape;
                let min = 2 + 2 * usize::from(*post_concat);
                let out = Shape::new(is.rows() / 2, is.cols());
                ret(is.rows() % min == 0, out, GraphIROpError::new(self, InvalidInputShape(is)))
            }
            PowerError(a, b, _) => {
                check_dense_eq(a, true)?;
                check_dense_eq(b, true)?;
                ret(a.shape == b.shape, a.shape, mismatch(&[a, b]))
            }
            ReduceAcrossBatch(node) => {
                check_dense_eq(node, true)?;
                let is = node.shape;
                ret(is == Shape::new(1, 1), is, GraphIROpError::new(self, InvalidInputShape(is)))
            }
            Select(input, buckets) => {
                check_dense_eq(input, true)?;
                check_dense_eq(buckets, false)?;
                let is = input.shape;
                let bs = buckets.shape;
                let valid = is.cols() == bs.cols() && is.rows() % bs.rows() == 0;
                let out = Shape::new(is.rows() / bs.rows(), is.cols());
                ret(valid, out, mismatch(&[input, buckets]))
            }
            Slice(input, start, end) => {
                check_dense_eq(input, true)?;
                let is = input.shape;
                let valid = end > start && *end <= is.rows() && is.cols() == 1;
                let out = Shape::new(end - start, 1);
                ret(valid, out, GraphIROpError::new(self, OutOfBounds(is, [*start, *end])))
            }
            SparseAffine(w, i, b) => {
                check_dense_eq(w, true)?;
                check_dense_eq(i, false)?;
                check_not_batched(w)?;

                if let Some(b) = b {
                    check_dense_eq(b, true)?;
                }

                let out = check_matmul(w.shape, i.shape)?;
                ret(b.is_none() || out == b.unwrap().shape, out, mismatch(&[w, i]))
            }
            SparseAffineDualActivate(w, s, n, b, act) => {
                check_dense_eq(w, true)?;
                check_dense_eq(b, true)?;
                check_dense_eq(s, false)?;
                check_dense_eq(n, false)?;
                check_not_batched(w)?;
                let shb = b.shape;

                if *act == Activation::Square {
                    return Err(GraphIROpError::new(self, GraphIROpErrorType::ActivationCannotBeFused));
                }

                let out = check_matmul(w.shape, s.shape)?;
                let valid = s.shape == n.shape && out == shb;
                ret(valid, Shape::new(2 * shb.rows(), shb.cols()), mismatch(&[w, s, n, b]))
            }
            ToDense(node) => {
                check_dense_eq(node, false)?;
                Ok(node.shape)
            }
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => {
                check_dense_eq(input, true)?;
                check_dense_eq(target, true)?;
                let is = input.shape;
                let valid = mask.sparse.unwrap().get() == target.shape.rows()
                    && mask.shape == is
                    && is.cols() == 1
                    && target.shape.cols() == 1;
                ret(valid, Shape::new(1, 1), mismatch(&[mask, input, target]))
            }
            SoftmaxCrossEntropyLoss(a, b) => {
                check_dense_eq(a, true)?;
                check_dense_eq(b, true)?;
                ret(a.shape == b.shape, Shape::new(1, 1), mismatch(&[a, b]))
            }
        }
    }

    pub fn nodes(&self) -> Vec<AnnotatedNode> {
        use GraphIROp::*;

        match *self {
            Activate(node, _) => vec![node],
            Affine(a, b, c) => vec![a, b, c],
            Concat(a, b) => vec![a, b],
            Gather(input, mask) => vec![input, mask],
            LinearCombination(_, a, _, b) => vec![a, b],
            Mask(input, mask) => vec![input, mask],
            Matmul(a, _, b, _) => vec![a, b],
            PairwiseMul(input, _) => vec![input],
            PowerError(a, b, _) => vec![a, b],
            ReduceAcrossBatch(node) => vec![node],
            Select(input, buckets) => vec![input, buckets],
            Slice(input, _, _) => vec![input],
            SparseAffine(w, i, b) => {
                if let Some(b) = b {
                    vec![w, i, b]
                } else {
                    vec![w, i]
                }
            }
            ToDense(node) => vec![node],
            SparseAffineDualActivate(w, s, n, b, _) => vec![w, s, n, b],
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => vec![mask, input, target],
            SoftmaxCrossEntropyLoss(a, b) => vec![a, b],
        }
    }
}
