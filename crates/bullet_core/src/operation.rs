use crate::{
    conv::ConvolutionDescription,
    device::Device,
    graph::{GraphBuilder, Node},
    shape::Shape,
    tensor::Tensor,
};

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
    AffineDual(Node, Node, Node, Node),
    Concat(Node, Node),
    Convolution(Node, Node, ConvolutionDescription),
    Gather(Node, Node),
    LinearCombination(f32, Node, f32, Node),
    Mask(Node, Node),
    PairwiseMul(Node, bool),
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

        let mismatch = |op: &Operation, nodes: &[&Node]| OperationError {
            op: Box::new(*op),
            ty: MismatchedInputShapes(nodes.iter().map(|&x| shape(x)).collect::<Vec<_>>()),
        };

        match self {
            Activate(node, _) => Ok(shape(node)),
            Affine(w, i, b) => {
                let valid = shape(w) * shape(i) == shape(b);
                ret(valid, shape(b), mismatch(self, &[w, i, b]))
            }
            AffineDual(w, s, n, b) => {
                let valid = shape(s) == shape(n) && shape(w) * shape(s) == shape(b);
                ret(valid, shape(b), mismatch(self, &[w, s, n, b]))
            }
            Concat(a, b) => {
                if shape(a).rows() != 1 {
                    return Err(OperationError::new(self, InvalidInputShape(shape(a))));
                }

                let out = Shape::new(shape(a).rows() + shape(b).rows(), shape(a).cols());
                ret(shape(a).cols() == shape(b).cols(), out, mismatch(self, &[a, b]))
            }
            Convolution(filters, inputs, desc) => {
                let valid = shape(inputs).cols() == 1
                    && shape(inputs).size() == desc.input_shape.size() * desc.input_channels
                    && shape(filters).rows() == desc.filter_shape.size()
                    && shape(filters).cols() == desc.input_channels * desc.output_channels;
                let out = Shape::new(desc.output_shape.size() * desc.output_channels, 1);
                ret(valid, out, mismatch(self, &[filters, inputs]))
            }
            Gather(input, mask) => {
                let valid = shape(input).cols() == 1 && shape(mask).cols() == 1;
                ret(valid, shape(mask), mismatch(self, &[input, mask]))
            }
            LinearCombination(_, a, _, b) => ret(shape(a) == shape(b), shape(a), mismatch(self, &[a, b])),
            Mask(input, mask) => ret(shape(input) == shape(mask), shape(input), mismatch(self, &[input, mask])),
            PairwiseMul(input, post_concat) => {
                let is = shape(input);
                let min = 2 + 2 * usize::from(*post_concat);
                let out = Shape::new(is.rows() / 2, is.cols());
                ret(is.rows() % min == 0, out, OperationError::new(self, InvalidInputShape(is)))
            }
            Select(input, buckets) => {
                let is = shape(input);
                let bs = shape(buckets);
                let valid = is.cols() == bs.cols() && is.rows() % bs.rows() == 0;
                let out = Shape::new(is.rows() / bs.rows(), is.cols());
                ret(valid, out, mismatch(self, &[input, buckets]))
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
                ret(valid, Shape::new(1, 1), mismatch(self, &[mask, input, target]))
            }
            SoftmaxCrossEntropyLoss(a, b) => ret(shape(a) == shape(b), Shape::new(1, 1), mismatch(self, &[a, b])),
            SubmatrixProduct(a, b, m) => {
                let ash = shape(a);
                let bsh = shape(b);
                let valid = ash.cols() == 1 && bsh.cols() == 1 && ash.rows() % m == 0 && bsh.rows() % m == 0;
                let inp1 = Shape::new(*m, ash.rows() / m);
                let inp2 = Shape::new(*m, bsh.rows() / m);
                let out = inp1.transpose() * inp2;
                ret(valid, Shape::new(out.size(), 1), mismatch(self, &[a, b]))
            }
        }
    }
}

type T<'a, D> = &'a Tensor<D>;
type Tm<'a, D> = &'a mut Tensor<D>;

pub enum OperationForward<'a, D: Device> {
    Activate(T<'a, D>, Activation),
    Affine(T<'a, D>, T<'a, D>, T<'a, D>),
    AffineDual(T<'a, D>, T<'a, D>, T<'a, D>, T<'a, D>),
    Concat(T<'a, D>, T<'a, D>),
    Convolution(T<'a, D>, T<'a, D>, ConvolutionDescription),
    Gather(T<'a, D>, T<'a, D>),
    LinearCombination(f32, T<'a, D>, f32, T<'a, D>),
    Mask(T<'a, D>, T<'a, D>),
    PairwiseMul(T<'a, D>, bool),
    Select(T<'a, D>, T<'a, D>),
    Slice(T<'a, D>, usize, usize),
    MaskedSoftmaxCrossEntropyLoss(T<'a, D>, T<'a, D>, T<'a, D>),
    SoftmaxCrossEntropyLoss(T<'a, D>, T<'a, D>),
    SubmatrixProduct(T<'a, D>, T<'a, D>, usize),
}

pub enum OperationBackward<'a, D: Device> {
    Activate(Tm<'a, D>, Activation),
    Affine(Tm<'a, D>, Tm<'a, D>, Tm<'a, D>),
    AffineDual(Tm<'a, D>, Tm<'a, D>, Tm<'a, D>, Tm<'a, D>),
    Concat(Tm<'a, D>, Tm<'a, D>),
    Convolution(Tm<'a, D>, Tm<'a, D>, ConvolutionDescription),
    Gather(Tm<'a, D>, Tm<'a, D>),
    LinearCombination(f32, Tm<'a, D>, f32, Tm<'a, D>),
    Mask(Tm<'a, D>, Tm<'a, D>),
    PairwiseMul(Tm<'a, D>, bool),
    Select(Tm<'a, D>, Tm<'a, D>),
    Slice(Tm<'a, D>, usize, usize),
    MaskedSoftmaxCrossEntropyLoss(Tm<'a, D>, Tm<'a, D>, Tm<'a, D>),
    SoftmaxCrossEntropyLoss(Tm<'a, D>, Tm<'a, D>),
    SubmatrixProduct(Tm<'a, D>, Tm<'a, D>, usize),
}

/*
impl<'a, D: Device> OperationExecute<'a, D> {
    pub fn from_op(op: &Operation, graph: &'a Graph<D>) -> Self {
        use Operation::*;

        let t = |node: &Node| graph.get_node(*node);

        match op {
            Activate(i, a) => Self::Activate(t(i), *a),
            Affine(w, i, b) => Self::Affine(t(w), t(i), t(b)),
            AffineDual(w, s, n, b) => Self::AffineDual(t(w), t(s), t(n), t(b))
            Concat(a, b) => Self::Concat(t(a), t(b)),
            Convolution(f, i, d) => Self::Convolution(t(f), t(i), *d),
            Gather(i, m) => Self::Gather(t(i), t(m)),
            LinearCombination(alpha, a, beta, b) => Self::LinearCombination(*alpha, t(a), *beta, t(b)),
            Mask(i, m) => Self::Mask(t(i), t(m)),
            PairwiseMul(x, b) => Self::PairwiseMul(t(x), *b),
            Select(i, b) => Self::Select(t(i), t(b)),
            Slice(x, s, e) => Self::Slice(t(x), *s, *e),
            MaskedSoftmaxCrossEntropyLoss(a, b, m) => Self::MaskedSoftmaxCrossEntropyLoss(t(a), t(b), t(m)),
            SoftmaxCrossEntropyLoss(a, b) => Self::SoftmaxCrossEntropyLoss(t(a), t(b)),
            SubmatrixProduct(a, b, m) => Self::SubmatrixProduct(t(a), t(b), *m),
        }
    }
}
*/
