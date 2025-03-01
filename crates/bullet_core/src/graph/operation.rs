pub mod concat;
pub mod linear_comb;
pub mod matmul;
pub mod slice;
pub mod sparse;

use std::{cell::RefCell, collections::HashMap, sync::Arc};

use crate::{device::Device, graph::Node, shape::Shape, tensor::DenseMatrix};

/// List of supported activation functions.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Identity = 0,
    ReLU = 1,
    CReLU = 2,
    SCReLU = 3,
    SqrReLU = 4,
    Sigmoid = 5,
    Square = 6,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operation {
    Activate(Node, Activation),
    Affine(Node, Node, Node),
    SparseAffine(Node, Node, Option<Node>),
    SparseAffineDualActivate(Node, Node, Node, Node, Activation),
    Concat(Node, Node),
    Gather(Node, Node),
    LinearCombination(f32, Node, f32, Node),
    Mask(Node, Node),
    Matmul(Node, bool, Node, bool),
    PairwiseMul(Node, bool),
    PowerError(Node, Node, f32),
    ReduceAcrossBatch(Node),
    Select(Node, Node),
    Slice(Node, usize, usize),
    ToDense(Node),
    MaskedSoftmaxCrossEntropyLoss(Node, Node, Node),
    SoftmaxCrossEntropyLoss(Node, Node),
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphBuilderError {
    pub op: Box<Operation>,
    pub ty: GraphBuilderErrorType,
}

impl GraphBuilderError {
    pub fn new(op: &Operation, ty: GraphBuilderErrorType) -> Self {
        Self { op: Box::new(*op), ty }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphBuilderErrorType {
    InvalidInputShape(Shape),
    MismatchedInputShapes(Vec<Shape>),
    OutOfBounds(Shape, [usize; 2]),
    IncorrectDataLayout,
    BatchedInputNotSupported,
    InvalidMatmulDims,
    ActivationCannotBeFused,
    NodeWithIdAlreadyExists,
}

impl Operation {
    pub fn output_shape(&self) -> Result<Shape, GraphBuilderError> {
        use GraphBuilderErrorType::*;
        use Operation::*;

        let ret = |cond, ok, err| if cond { Ok(ok) } else { Err(err) };

        let mismatch = |nodes: &[&Node]| GraphBuilderError {
            op: Box::new(*self),
            ty: MismatchedInputShapes(nodes.iter().map(|&x| x.shape).collect::<Vec<_>>()),
        };

        let check_dense_eq = |node: &Node, dense: bool| {
            if node.sparse.is_none() == dense {
                Ok(())
            } else {
                Err(GraphBuilderError::new(self, GraphBuilderErrorType::IncorrectDataLayout))
            }
        };

        let check_not_batched = |node: &Node| {
            if node.can_be_batched {
                Err(GraphBuilderError::new(self, GraphBuilderErrorType::BatchedInputNotSupported))
            } else {
                Ok(())
            }
        };

        let check_matmul = |a: Shape, b: Shape| {
            if let Some(c) = a.matmul(b) {
                Ok(c)
            } else {
                Err(GraphBuilderError::new(self, GraphBuilderErrorType::InvalidMatmulDims))
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
                    return Err(GraphBuilderError::new(self, InvalidInputShape(a.shape)));
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
                ret(is.rows() % min == 0, out, GraphBuilderError::new(self, InvalidInputShape(is)))
            }
            PowerError(a, b, _) => {
                check_dense_eq(a, true)?;
                check_dense_eq(b, true)?;
                ret(a.shape == b.shape, a.shape, mismatch(&[a, b]))
            }
            ReduceAcrossBatch(node) => {
                check_dense_eq(node, true)?;
                let is = node.shape;
                ret(is == Shape::new(1, 1), is, GraphBuilderError::new(self, InvalidInputShape(is)))
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
                ret(valid, out, GraphBuilderError::new(self, OutOfBounds(is, [*start, *end])))
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
                    return Err(GraphBuilderError::new(self, GraphBuilderErrorType::ActivationCannotBeFused));
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

    pub fn nodes(&self) -> Vec<Node> {
        use Operation::*;

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

pub fn setup_ones<D: Device>(
    device: Arc<D>,
    internal: &mut HashMap<String, RefCell<DenseMatrix<D>>>,
    batch_size: usize,
) -> Result<(), D::DeviceError> {
    if let Some(ones) = internal.get_mut("ones") {
        if ones.borrow().size() < batch_size {
            *ones = RefCell::new(DenseMatrix::ones(device, batch_size)?);
        }
    } else {
        let ones = RefCell::new(DenseMatrix::ones(device, batch_size)?);
        internal.insert("ones".to_string(), ones);
    }

    Ok(())
}

pub fn setup_softmax<D: Device>(
    device: Arc<D>,
    internal: &mut HashMap<String, RefCell<DenseMatrix<D>>>,
    size: usize,
) -> Result<(), D::DeviceError> {
    if !internal.contains_key("softmaxed") {
        let zeros = RefCell::new(DenseMatrix::zeroed(device.clone(), size)?);
        internal.insert("softmaxed".to_string(), zeros);
        let zeros = RefCell::new(DenseMatrix::zeroed(device, size)?);
        internal.insert("individual_losses".to_string(), zeros);
    }

    Ok(())
}
