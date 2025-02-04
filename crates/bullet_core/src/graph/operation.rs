mod concat;
mod conv;
mod matmul;

use std::{cell::RefCell, collections::HashMap, ops::Deref, sync::Arc};

use crate::{
    device::{Device, DeviceBuffer},
    graph::{builder::GraphBuilder, Node},
    shape::Shape,
    tensor::DenseMatrix,
};

pub use conv::ConvolutionDescription;

use super::Graph;

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
}

#[derive(Clone, Copy, Debug)]
pub enum Operation {
    Activate(Node, Activation),
    Affine(Node, Node, Node),
    AffineDualActivate(Node, Node, Node, Node, Activation),
    Concat(Node, Node),
    Convolution(Node, Node, ConvolutionDescription),
    Gather(Node, Node),
    LinearCombination(f32, Node, f32, Node),
    Linear(Node, Node),
    Mask(Node, Node),
    PairwiseMul(Node, bool),
    PowerError(Node, Node, f32),
    ReduceAcrossBatch(Node),
    Select(Node, Node),
    Slice(Node, usize, usize),
    MaskedSoftmaxCrossEntropyLoss(Node, Node, Node),
    SoftmaxCrossEntropyLoss(Node, Node),
    SubmatrixProduct(Node, Node, usize),
}

#[derive(Clone, Debug)]
pub struct OperationError {
    pub op: Box<Operation>,
    pub ty: OperationErrorType,
}

impl OperationError {
    pub fn new(op: &Operation, ty: OperationErrorType) -> Self {
        Self { op: Box::new(*op), ty }
    }
}

#[derive(Clone, Debug)]
pub enum OperationErrorType {
    InvalidInputShape(Shape),
    MismatchedInputShapes(Vec<Shape>),
    OutOfBounds(Shape, [usize; 2]),
}

impl Operation {
    pub fn output_shape<D: Device>(&self, builder: &GraphBuilder<D>) -> Result<Shape, OperationError> {
        use Operation::*;
        use OperationErrorType::*;

        let shape = |node: &Node| builder.get_node(*node).shape();

        let ret = |cond, ok, err| if cond { Ok(ok) } else { Err(err) };

        let mismatch = |nodes: &[&Node]| OperationError {
            op: Box::new(*self),
            ty: MismatchedInputShapes(nodes.iter().map(|&x| shape(x)).collect::<Vec<_>>()),
        };

        match self {
            Activate(node, _) => Ok(shape(node)),
            Affine(w, i, b) => {
                let valid = shape(w) * shape(i) == shape(b);
                ret(valid, shape(b), mismatch(&[w, i, b]))
            }
            AffineDualActivate(w, s, n, b, _) => {
                let valid = shape(s) == shape(n) && shape(w) * shape(s) == shape(b);
                ret(valid, shape(b), mismatch(&[w, s, n, b]))
            }
            Concat(a, b) => {
                if shape(a).rows() != 1 {
                    return Err(OperationError::new(self, InvalidInputShape(shape(a))));
                }

                let out = Shape::new(shape(a).rows() + shape(b).rows(), shape(a).cols());
                ret(shape(a).cols() == shape(b).cols(), out, mismatch(&[a, b]))
            }
            Convolution(filters, inputs, desc) => {
                let valid = shape(inputs).cols() == 1
                    && shape(inputs).size() == desc.input_shape.size() * desc.input_channels
                    && shape(filters).rows() == desc.filter_shape.size()
                    && shape(filters).cols() == desc.input_channels * desc.output_channels;
                let out = Shape::new(desc.output_shape.size() * desc.output_channels, 1);
                ret(valid, out, mismatch(&[filters, inputs]))
            }
            Gather(input, mask) => {
                let valid = shape(input).cols() == 1 && shape(mask).cols() == 1;
                ret(valid, shape(mask), mismatch(&[input, mask]))
            }
            LinearCombination(_, a, _, b) => ret(shape(a) == shape(b), shape(a), mismatch(&[a, b])),
            Linear(a, b) => ret(true, shape(a) * shape(b), mismatch(&[a, b])),
            Mask(input, mask) => ret(shape(input) == shape(mask), shape(input), mismatch(&[input, mask])),
            PairwiseMul(input, post_concat) => {
                let is = shape(input);
                let min = 2 + 2 * usize::from(*post_concat);
                let out = Shape::new(is.rows() / 2, is.cols());
                ret(is.rows() % min == 0, out, OperationError::new(self, InvalidInputShape(is)))
            }
            PowerError(a, b, _) => ret(shape(a) == shape(b), shape(a), mismatch(&[a, b])),
            ReduceAcrossBatch(node) => {
                let is = shape(node);
                ret(is == Shape::new(1, 1), is, OperationError::new(self, InvalidInputShape(is)))
            }
            Select(input, buckets) => {
                let is = shape(input);
                let bs = shape(buckets);
                let valid = is.cols() == bs.cols() && is.rows() % bs.rows() == 0;
                let out = Shape::new(is.rows() / bs.rows(), is.cols());
                ret(valid, out, mismatch(&[input, buckets]))
            }
            Slice(input, start, end) => {
                let is = shape(input);
                let valid = end > start && *end <= is.rows() && is.cols() == 1;
                let out = Shape::new(end - start, 1);
                ret(valid, out, OperationError::new(self, OutOfBounds(is, [*start, *end])))
            }
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => {
                let is = shape(input);
                let valid = shape(mask) == is && is.cols() == 1 && shape(target).cols() == 1;
                ret(valid, Shape::new(1, 1), mismatch(&[mask, input, target]))
            }
            SoftmaxCrossEntropyLoss(a, b) => ret(shape(a) == shape(b), Shape::new(1, 1), mismatch(&[a, b])),
            SubmatrixProduct(a, b, m) => {
                let ash = shape(a);
                let bsh = shape(b);
                let valid = ash.cols() == 1 && bsh.cols() == 1 && ash.rows() % m == 0 && bsh.rows() % m == 0;
                let out = Shape::new(*m, ash.rows() / m).transpose() * Shape::new(*m, bsh.rows() / m);
                ret(valid, Shape::new(out.size(), 1), mismatch(&[a, b]))
            }
        }
    }
}

impl<D: Device> Graph<D> {
    pub fn forward_node(&mut self, output_node: Node, op: &Operation) {
        use Operation::*;

        let get = |node: Node| self.nodes[node.0].borrow();

        let output_tensor = &mut *self.nodes[output_node.0].borrow_mut();
        let internal = &mut output_tensor.internal;
        let output = output_tensor.values.dense_mut();

        match op {
            Activate(node, act) => D::activate(get(*node).values.dense(), output, *act),
            Affine(w, i, b) => {
                let i = get(*i);
                let w = get(*w);
                let w = w.values.dense();
                let bs = i.values.shape().batch_size().unwrap_or(1);
                setup_ones(w.buf.device(), internal, bs);
                let ones = &internal.get("ones").unwrap().borrow().buf;
                matmul::linear(w, &*i, output);
                D::add_assign_single_to_batched_scaled(ones, 1.0, get(*b).values.dense(), output);
            }
            AffineDualActivate(w, s, n, b, act) => {
                D::sparse_affine_dual_activate(
                    get(*w).values.dense(),
                    get(*s).values.sparse(),
                    get(*n).values.sparse(),
                    get(*b).values.dense(),
                    output,
                    *act,
                );
            }
            LinearCombination(alpha, a, beta, b) => {
                let a = get(*a);
                let a = a.values.dense();
                let bs = a.shape().batch_size().unwrap_or(1);
                setup_ones(a.buf.device(), internal, bs);
                let ones = &internal.get("ones").unwrap().borrow().buf;
                D::linear_comb(ones, *alpha, a, *beta, get(*b).values.dense(), output);
            }
            Convolution(filters, inputs, desc) => {
                D::convolution_forward(desc, get(*filters).values.dense(), get(*inputs).values.dense(), output)
            }
            Gather(input, mask) => D::gather(get(*input).values.dense(), get(*mask).values.sparse(), output),
            Concat(a, b) => concat::concat(get(*a).values.dense(), get(*b).values.dense(), output),
            Linear(a, b) => matmul::linear(get(*a).values.dense(), get(*b).deref(), output),
            Mask(input, mask) => D::mask(get(*input).values.dense(), get(*mask).values.sparse(), output),
            PairwiseMul(input, post_concat) => D::pairwise(get(*input).values.dense(), output, *post_concat),
            PowerError(a, b, p) => D::power_error(*p, get(*a).values.dense(), get(*b).values.dense(), output),
            ReduceAcrossBatch(input) => {
                let input = get(*input);
                let input = input.values.dense();
                setup_ones(input.buf.device(), internal, input.shape().batch_size().unwrap_or(1));
                let ones = internal.get("ones").unwrap().borrow();
                D::reduce_add_batch(&ones.buf, input, output);
            }
            Select(input, buckets) => D::select(get(*input).values.dense(), get(*buckets).values.sparse(), output),
            Slice(input, start, end) => D::slice_vector_batched(get(*input).values.dense(), *start, *end, output),
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => {
                let input = get(*input);
                let input = input.values.dense();
                let mask = get(*mask);
                let mask = mask.values.sparse();
                setup_softmax(input.buf.device(), internal, input.shape());
                let mut smax = internal.get("softmaxed").unwrap().borrow_mut();
                let mut indv = internal.get("individual_losses").unwrap().borrow_mut();
                D::softmax_across_batch_masked(mask, input, &mut smax);
                D::crossentropy_loss_masked(mask, &smax, get(*target).values.dense(), &mut indv, output);
            }
            SoftmaxCrossEntropyLoss(a, b) => {
                let a = get(*a);
                let a = a.values.dense();
                setup_softmax(a.buf.device(), internal, a.shape());
                setup_ones(a.buf.device(), internal, a.shape().size());
                let ones = internal.get("ones").unwrap().borrow();
                let mut smax = internal.get("softmaxed").unwrap().borrow_mut();
                let mut indv = internal.get("individual_losses").unwrap().borrow_mut();
                D::softmax_across_batch(a, &mut smax);
                D::crossentropy_loss(&ones.buf, &smax, get(*b).values.dense(), &mut indv, output);
            }
            SubmatrixProduct(a, b, m) => {
                matmul::submatrix_product(*m, get(*a).values.dense(), get(*b).values.dense(), output)
            }
        }
    }
}

fn setup_ones<D: Device>(device: Arc<D>, internal: &mut HashMap<String, RefCell<DenseMatrix<D>>>, batch_size: usize) {
    if let Some(ones) = internal.get_mut("ones") {
        if ones.borrow().shape.size() < batch_size {
            *ones = RefCell::new(DenseMatrix::ones(device, Shape::new(batch_size, 1)));
        }
    } else {
        let ones = RefCell::new(DenseMatrix::ones(device, Shape::new(batch_size, 1)));
        internal.insert("ones".to_string(), ones);
    }
}

fn setup_softmax<D: Device>(device: Arc<D>, internal: &mut HashMap<String, RefCell<DenseMatrix<D>>>, shape: Shape) {
    if !internal.contains_key("softmaxed") {
        let zeros = RefCell::new(DenseMatrix::zeroed(device.clone(), shape));
        internal.insert("softmaxed".to_string(), zeros);
        let zeros = RefCell::new(DenseMatrix::zeroed(device, shape));
        internal.insert("individual_losses".to_string(), zeros);
    }
}
