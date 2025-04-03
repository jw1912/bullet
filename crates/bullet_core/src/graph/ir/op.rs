use super::{node::AnnotatedNode, GraphIR, GraphIRError, Shape};

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
pub enum GraphIROp {
    Affine(AnnotatedNode, AnnotatedNode, AnnotatedNode),
    SparseAffineActivate(AnnotatedNode, AnnotatedNode, Option<AnnotatedNode>, DiffableFromOutput),
    SparseAffineDualActivate(AnnotatedNode, AnnotatedNode, AnnotatedNode, Option<AnnotatedNode>, DiffableFromOutput),
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
    Unary(AnnotatedNode, UnaryOp),
    MaskedSoftmaxCrossEntropyLoss(AnnotatedNode, AnnotatedNode, AnnotatedNode),
    SoftmaxCrossEntropyLoss(AnnotatedNode, AnnotatedNode),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnaryOp {
    DiffableFromOutput(DiffableFromOutput),
    Add(f32),
    Mul(f32),
    AbsPow(f32),
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
    AnnotatedNodeWithIdAlreadyExists,
    MismatchedBatching,
}

impl GraphIROp {
    pub fn output_info(&self, ir: &GraphIR) -> Result<(Shape, bool, bool), GraphIRError> {
        use GraphIROp::*;
        use GraphIROpErrorType::*;

        let ret = |cond, ok, err| if cond { Ok(ok) } else { Err(err) };

        let get = |node: &AnnotatedNode| ir.get(node.idx).unwrap();

        let mismatch = |nodes: &[&AnnotatedNode]| GraphIROpError {
            op: Box::new(*self),
            ty: MismatchedInputShapes(nodes.iter().map(|&x| x.shape).collect::<Vec<_>>()),
        };

        let check_dense_eq = |node: &AnnotatedNode, dense: bool| {
            if get(node).sparse.is_none() == dense {
                Ok(())
            } else {
                Err(GraphIROpError::new(self, GraphIROpErrorType::IncorrectDataLayout))
            }
        };

        let check_not_batched = |node: &AnnotatedNode| {
            if get(node).batched {
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

        let check_same_batching = |x: &[&AnnotatedNode]| {
            if x.iter().all(|y| get(y).batched == get(x[0]).batched) {
                Ok(())
            } else {
                Err(GraphIROpError::new(self, GraphIROpErrorType::MismatchedBatching))
            }
        };

        for node in self.nodes() {
            if node.shape.size() != get(&node).shape.size() {
                let err = GraphIROpError::new(self, GraphIROpErrorType::InvalidInputShape(node.shape));
                return Err(GraphIRError::Op(err));
            }
        }

        let mut batched = self.nodes().iter().any(|node| get(node).batched);

        let requires_grad = self.nodes().iter().any(|node| get(node).requires_grad);

        let shape = match self {
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
                check_same_batching(&[a, b])?;

                if a.shape.cols() != 1 {
                    return Err(GraphIRError::Op(GraphIROpError::new(self, InvalidInputShape(a.shape))));
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
                check_dense_eq(input, true)?;
                let is = input.shape;
                let min = 2 + 2 * usize::from(*post_concat);
                let out = Shape::new(is.rows() / 2, is.cols());
                ret(is.rows() % min == 0, out, GraphIROpError::new(self, InvalidInputShape(is)))
            }
            PowerError(a, b, _) => {
                check_dense_eq(a, true)?;
                check_dense_eq(b, true)?;
                check_same_batching(&[a, b])?;
                ret(a.shape == b.shape, a.shape, mismatch(&[a, b]))
            }
            ReduceAcrossBatch(node) => {
                check_dense_eq(node, true)?;
                let is = node.shape;
                batched = false;
                ret(is == Shape::new(1, 1), is, GraphIROpError::new(self, InvalidInputShape(is)))
            }
            Select(input, buckets) => {
                check_dense_eq(input, true)?;
                check_dense_eq(buckets, false)?;
                check_same_batching(&[input, buckets])?;
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
            SparseAffineActivate(w, i, b, _) => {
                check_dense_eq(w, true)?;
                check_dense_eq(i, false)?;
                check_not_batched(w)?;

                if let Some(b) = b {
                    check_dense_eq(b, true)?;
                }

                let out = check_matmul(w.shape, i.shape)?;
                ret(b.is_none() || out == b.unwrap().shape, out, mismatch(&[w, i]))
            }
            SparseAffineDualActivate(w, s, n, b, _) => {
                check_dense_eq(w, true)?;
                check_dense_eq(s, false)?;
                check_dense_eq(n, false)?;
                check_not_batched(w)?;
                check_same_batching(&[s, n])?;

                let out = check_matmul(w.shape, s.shape)?;
                let mut valid = s.shape == n.shape;

                if let Some(b) = b {
                    check_dense_eq(b, true)?;
                    valid &= out == b.shape
                }

                ret(valid, Shape::new(2 * out.rows(), out.cols()), mismatch(&[w, s, n]))
            }
            ToDense(node) => {
                check_dense_eq(node, false)?;
                Ok(node.shape)
            }
            Unary(node, _) => {
                check_dense_eq(node, true)?;
                Ok(node.shape)
            }
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => {
                check_dense_eq(input, true)?;
                check_dense_eq(target, true)?;
                let is = input.shape;
                let valid = get(mask).sparse.unwrap().get() == target.shape.rows()
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
        }?;

        Ok((shape, batched, requires_grad))
    }

    pub fn nodes(&self) -> Vec<AnnotatedNode> {
        use GraphIROp::*;

        match *self {
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
            SparseAffineActivate(w, i, b, _) => {
                if let Some(b) = b {
                    vec![w, i, b]
                } else {
                    vec![w, i]
                }
            }
            ToDense(node) => vec![node],
            Unary(node, _) => vec![node],
            SparseAffineDualActivate(w, s, n, b, _) => {
                if let Some(b) = b {
                    vec![w, s, n, b]
                } else {
                    vec![w, s, n]
                }
            }
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => vec![mask, input, target],
            SoftmaxCrossEntropyLoss(a, b) => vec![a, b],
        }
    }
}
