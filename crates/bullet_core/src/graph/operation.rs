mod concat;
mod conv;
mod matmul;
mod slice;

use std::{cell::RefCell, collections::HashMap, sync::Arc};

use crate::{
    device::{Device, DeviceBuffer},
    graph::Node,
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
    Affine(Node, Node, Option<Node>),
    AffineDualActivate(Node, Node, Node, Node, Activation),
    Concat(Node, Node),
    Convolution(Node, Node, ConvolutionDescription),
    Gather(Node, Node),
    LinearCombination(f32, Node, f32, Node),
    Mask(Node, Node),
    PairwiseMul(Node, bool),
    PowerError(Node, Node, f32),
    ReduceAcrossBatch(Node),
    Select(Node, Node),
    Slice(Node, usize, usize),
    MaskedSoftmaxCrossEntropyLoss(Node, Node, Node),
    SoftmaxCrossEntropyLoss(Node, Node),
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
    pub fn output_shape(&self) -> Result<Shape, OperationError> {
        use Operation::*;
        use OperationErrorType::*;

        let ret = |cond, ok, err| if cond { Ok(ok) } else { Err(err) };

        let mismatch = |nodes: &[&Node]| OperationError {
            op: Box::new(*self),
            ty: MismatchedInputShapes(nodes.iter().map(|&x| x.shape).collect::<Vec<_>>()),
        };

        match self {
            Activate(node, _) => Ok(node.shape),
            Affine(w, i, b) => {
                let out = w.shape * i.shape;
                let valid = b.is_none() || out == b.unwrap().shape;
                ret(valid, out, mismatch(&[w, i]))
            }
            AffineDualActivate(w, s, n, b, _) => {
                let shb = b.shape;
                let valid = s.shape == n.shape && w.shape * s.shape == shb;
                ret(valid, Shape::new(2 * shb.rows(), shb.cols()), mismatch(&[w, s, n, b]))
            }
            Concat(a, b) => {
                if a.shape.rows() != 1 {
                    return Err(OperationError::new(self, InvalidInputShape(a.shape)));
                }

                let out = Shape::new(a.shape.rows() + b.shape.rows(), a.shape.cols());
                ret(a.shape.cols() == b.shape.cols(), out, mismatch(&[a, b]))
            }
            Convolution(filters, inputs, desc) => {
                let valid = inputs.shape.cols() == 1
                    && inputs.shape.size() == desc.input_shape.size() * desc.input_channels
                    && filters.shape.rows() == desc.filter_shape.size()
                    && filters.shape.cols() == desc.input_channels * desc.output_channels;
                let out = Shape::new(desc.output_shape.size() * desc.output_channels, 1);
                ret(valid, out, mismatch(&[filters, inputs]))
            }
            Gather(input, mask) => {
                let valid = input.shape.cols() == 1 && mask.shape.cols() == 1;
                ret(valid, mask.shape, mismatch(&[input, mask]))
            }
            LinearCombination(_, a, _, b) => ret(a.shape == b.shape, a.shape, mismatch(&[a, b])),
            Mask(input, mask) => ret(input.shape == mask.shape, input.shape, mismatch(&[input, mask])),
            PairwiseMul(input, post_concat) => {
                let is = input.shape;
                let min = 2 + 2 * usize::from(*post_concat);
                let out = Shape::new(is.rows() / 2, is.cols());
                ret(is.rows() % min == 0, out, OperationError::new(self, InvalidInputShape(is)))
            }
            PowerError(a, b, _) => ret(a.shape == b.shape, a.shape, mismatch(&[a, b])),
            ReduceAcrossBatch(node) => {
                let is = node.shape;
                ret(is == Shape::new(1, 1), is, OperationError::new(self, InvalidInputShape(is)))
            }
            Select(input, buckets) => {
                let is = input.shape;
                let bs = buckets.shape;
                let valid = is.cols() == bs.cols() && is.rows() % bs.rows() == 0;
                let out = Shape::new(is.rows() / bs.rows(), is.cols());
                ret(valid, out, mismatch(&[input, buckets]))
            }
            Slice(input, start, end) => {
                let is = input.shape;
                let valid = end > start && *end <= is.rows() && is.cols() == 1;
                let out = Shape::new(end - start, 1);
                ret(valid, out, OperationError::new(self, OutOfBounds(is, [*start, *end])))
            }
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => {
                let is = input.shape;
                let valid = mask.shape == is && is.cols() == 1 && target.shape.cols() == 1;
                ret(valid, Shape::new(1, 1), mismatch(&[mask, input, target]))
            }
            SoftmaxCrossEntropyLoss(a, b) => ret(a.shape == b.shape, Shape::new(1, 1), mismatch(&[a, b])),
        }
    }

    pub fn nodes(&self) -> Vec<Node> {
        use Operation::*;

        match *self {
            Activate(node, _) => vec![node],
            Affine(w, i, b) => {
                if let Some(b) = b {
                    vec![w, i, b]
                } else {
                    vec![w, i]
                }
            }
            AffineDualActivate(w, s, n, b, _) => vec![w, s, n, b],
            Concat(a, b) => vec![a, b],
            Convolution(filters, inputs, _) => vec![filters, inputs],
            Gather(input, mask) => vec![input, mask],
            LinearCombination(_, a, _, b) => vec![a, b],
            Mask(input, mask) => vec![input, mask],
            PairwiseMul(input, _) => vec![input],
            PowerError(a, b, _) => vec![a, b],
            ReduceAcrossBatch(node) => vec![node],
            Select(input, buckets) => vec![input, buckets],
            Slice(input, _, _) => vec![input],
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => vec![mask, input, target],
            SoftmaxCrossEntropyLoss(a, b) => vec![a, b],
        }
    }
}

impl<D: Device> Graph<D> {
    pub fn forward_node(&mut self, output_node: Node) {
        use Operation::*;

        let get = |node: Node| self.nodes[node.idx].borrow();

        let output_tensor = &mut *self.nodes[output_node.idx].borrow_mut();
        let op = if let Some(op) = &output_tensor.operation { op } else { return };
        let internal = &mut output_tensor.internal;
        let output = output_tensor.values.dense_mut();
        let outn = output_tensor.own;

        match op {
            Activate(node, act) => {
                let input = get(*node);
                let input = input.values.dense();
                assert_eq!(outn.shape, node.shape);
                output.set_batch_size(input.batch_size());
                D::activate(input.size(), &input.buf, &mut output.buf, *act);
            }
            Affine(wn, inp, bn) => {
                let i = get(*inp);
                let w = get(*wn);
                let w = w.values.dense();

                if let Some(b) = bn {
                    let bs = i.values.batch_size().unwrap_or(1);
                    setup_ones(w.buf.device(), internal, bs);
                    let ones = &internal.get("ones").unwrap().borrow().buf;
                    matmul::affine(w, wn.shape, &*i, inp.shape, Some((get(*b).values.dense(), ones)), output);
                } else {
                    matmul::affine(w, wn.shape, &*i, inp.shape, None, output);
                }
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
                let bs = a.batch_size().unwrap_or(1);
                setup_ones(a.buf.device(), internal, bs);
                let ones = &internal.get("ones").unwrap().borrow().buf;
                D::linear_comb(ones, *alpha, a, *beta, get(*b).values.dense(), output);
            }
            Convolution(filters, inputs, desc) => {
                D::convolution_forward(desc, get(*filters).values.dense(), get(*inputs).values.dense(), output)
            }
            Gather(input, mask) => D::gather(get(*input).values.dense(), get(*mask).values.sparse(), output),
            Concat(a, b) => concat::concat(get(*a).values.dense(), a.shape, get(*b).values.dense(), b.shape, output),
            Mask(input, mask) => D::mask(get(*input).values.dense(), get(*mask).values.sparse(), output),
            PairwiseMul(input, post_concat) => D::pairwise(get(*input).values.dense(), output, *post_concat),
            PowerError(a, b, p) => D::power_error(*p, get(*a).values.dense(), get(*b).values.dense(), output),
            ReduceAcrossBatch(input) => {
                let input = get(*input);
                let input = input.values.dense();
                setup_ones(input.buf.device(), internal, input.batch_size().unwrap_or(1));
                let ones = internal.get("ones").unwrap().borrow();
                D::reduce_add_batch(&ones.buf, input, output);
            }
            Select(input, buckets) => D::select(get(*input).values.dense(), get(*buckets).values.sparse(), output),
            Slice(input, start, end) => {
                slice::slice_vector_batched(input.shape, get(*input).values.dense(), *start, *end, output)
            }
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => {
                let input = get(*input);
                let input = input.values.dense();
                let mask = get(*mask);
                let mask = mask.values.sparse();
                setup_softmax(input.buf.device(), internal, input.single_size());
                let mut smax = internal.get("softmaxed").unwrap().borrow_mut();
                let mut indv = internal.get("individual_losses").unwrap().borrow_mut();
                D::softmax_across_batch_masked(mask, input, &mut smax);
                D::crossentropy_loss_masked(mask, &smax, get(*target).values.dense(), &mut indv, output);
            }
            SoftmaxCrossEntropyLoss(an, bn) => {
                let a = get(*an);
                let a = a.values.dense();
                setup_softmax(a.buf.device(), internal, a.single_size());
                setup_ones(a.buf.device(), internal, an.shape.size());
                let ones = internal.get("ones").unwrap().borrow();
                let mut smax = internal.get("softmaxed").unwrap().borrow_mut();
                let mut indv = internal.get("individual_losses").unwrap().borrow_mut();
                D::softmax_across_batch(a, &mut smax);
                D::crossentropy_loss(&ones.buf, &smax, get(*bn).values.dense(), &mut indv, output);
            }
        }
    }
}

impl<D: Device> Graph<D> {
    pub fn backward_node(&mut self, output_node: Node) {
        use Operation::*;

        let get = |node: Node| self.nodes[node.idx].borrow_mut();

        let output_tensor = &mut *self.nodes[output_node.idx].borrow_mut();
        let op = if let Some(op) = &output_tensor.operation { op } else { return };
        let internal = &mut output_tensor.internal;
        let outn = output_tensor.own;
        let output_grad = if let Some(grad) = output_tensor.gradients.as_ref() {
            grad
        } else {
            return;
        };

        match op {
            Activate(node, act) => {
                let input = &mut *get(*node);
                if let Some(grad) = input.gradients.as_mut() {
                    let input = input.values.dense();
                    assert_eq!(outn.shape, node.shape);
                    assert_eq!(output_grad.size(), input.size());
                    assert_eq!(output_grad.batch_size(), input.batch_size());
                    grad.set_batch_size(output_grad.batch_size());
                    D::backprop_activate(input.size(), &input.buf, &mut grad.buf, &output_grad.buf, *act);
                }
            }
            Affine(wn, inp, bn) => {
                let i = &mut *get(*inp);
                let w = &mut *get(*wn);
                let o = output_tensor.values.dense();

                if let Some(b) = bn {
                    let bs = i.values.batch_size().unwrap_or(1);
                    setup_ones(w.values.dense().buf.device(), internal, bs);
                    let ones = &internal.get("ones").unwrap().borrow().buf;
                    matmul::backprop_affine(w, wn.shape, i, inp.shape, Some((&mut *get(*b), ones)), o, output_grad);
                } else {
                    matmul::backprop_affine(w, wn.shape, i, inp.shape, None, o, output_grad);
                }
            }
            AffineDualActivate(w, s, n, b, act) => {
                let w = &mut *get(*w);
                let b = &mut *get(*b);
                D::backprop_sparse_affine_dual_activate(
                    w.values.dense(),
                    w.gradients.as_mut().unwrap(),
                    get(*s).values.sparse(),
                    get(*n).values.sparse(),
                    b.values.dense(),
                    b.gradients.as_mut().unwrap(),
                    output_tensor.values.dense(),
                    output_grad,
                    *act,
                );
            }
            LinearCombination(alpha, an, beta, bn) => {
                let a = &mut *get(*an);
                let b = &mut *get(*bn);

                let abs = a.values.batch_size().unwrap_or(1);
                let bbs = b.values.batch_size().unwrap_or(1);
                let bs = abs.max(bbs);
                setup_ones(a.values.dense().buf.device(), internal, bs);
                let ones = &internal.get("ones").unwrap().borrow().buf;

                if let Some(grd) = a.gradients.as_mut() {
                    D::backprop_add_single_scaled(ones, *alpha, a.values.dense(), grd, output_grad);
                }

                if let Some(grd) = b.gradients.as_mut() {
                    D::backprop_add_single_scaled(ones, *beta, b.values.dense(), grd, output_grad);
                }
            }
            Convolution(filters, inputs, desc) => {
                let filters = &mut *get(*filters);
                let inputs = &mut *get(*inputs);
                D::convolution_backward(
                    desc,
                    filters.values.dense(),
                    filters.gradients.as_mut(),
                    inputs.values.dense(),
                    inputs.gradients.as_mut(),
                    output_grad,
                );
            }
            Gather(input, mask) => {
                let input = &mut *get(*input);
                let mask = &mut *get(*mask);

                if let Some(grd) = input.gradients.as_mut() {
                    D::backprop_gather(output_grad, mask.values.sparse(), input.values.dense(), grd);
                }
            }
            Concat(an, bn) => {
                let a = &mut *get(*an);
                let b = &mut *get(*bn);
                concat::backprop_concat(
                    a.values.dense(),
                    a.gradients.as_mut(),
                    an.shape,
                    b.values.dense(),
                    b.gradients.as_mut(),
                    bn.shape,
                    output_grad,
                );
            }
            Mask(input, mask) => {
                if let Some(grd) = get(*input).gradients.as_mut() {
                    D::backprop_mask(output_grad, get(*mask).values.sparse(), grd);
                }
            }
            PairwiseMul(input, post_concat) => {
                let input = &mut *get(*input);
                if let Some(grd) = input.gradients.as_mut() {
                    D::backprop_pairwise(input.values.dense(), output_grad, grd, *post_concat);
                }
            }
            PowerError(a, b, p) => {
                let a = &mut *get(*a);
                let b = &mut *get(*b);

                if let Some(grd) = a.gradients.as_mut() {
                    D::backprop_abs_power_error_single(*p, a.values.dense(), b.values.dense(), output_grad, grd);
                }

                if let Some(grd) = b.gradients.as_mut() {
                    D::backprop_abs_power_error_single(*p, b.values.dense(), a.values.dense(), output_grad, grd);
                }
            }
            ReduceAcrossBatch(input) => {
                let input = &mut *get(*input);
                if let Some(grd) = input.gradients.as_mut() {
                    let vals = input.values.dense();
                    let bs = vals.batch_size();
                    let ss = vals.single_size();

                    setup_ones(vals.buf.device(), internal, bs.unwrap_or(1));
                    let ones = &internal.get("ones").unwrap().borrow().buf;

                    assert!(output_grad.batch_size().is_none());
                    assert_eq!(vals.single_size(), output_grad.single_size());
                    assert_eq!(vals.single_size(), grd.single_size());

                    grd.set_batch_size(bs);
                    D::add_assign_single_to_batched_scaled(
                        ss,
                        bs.unwrap_or(1),
                        ones,
                        1.0,
                        &output_grad.buf,
                        &mut grd.buf,
                    );
                }
            }
            Select(input, buckets) => {
                let input = &mut *get(*input);
                if let Some(grd) = input.gradients.as_mut() {
                    D::select_backprop(input.values.dense(), get(*buckets).values.sparse(), output_grad, grd);
                }
            }
            Slice(node, start, end) => {
                let input = &mut *get(*node);
                if let Some(grd) = input.gradients.as_mut() {
                    slice::backprop_slice_vector_batched(
                        node.shape,
                        input.values.dense(),
                        grd,
                        *start,
                        *end,
                        output_grad,
                    );
                }
            }
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => {
                let mask = &*get(*mask);
                let mask = mask.values.sparse();
                let input = &mut *get(*input);
                let target = &mut *get(*target);

                let smax = internal.get("softmaxed").unwrap().borrow();

                if let Some(grd) = input.gradients.as_mut() {
                    D::backprop_softmax_crossentropy_loss_masked(mask, &smax, target.values.dense(), output_grad, grd);
                }

                if let Some(grd) = target.gradients.as_mut() {
                    D::backprop_softmax_crossentropy_loss_masked(mask, &smax, input.values.dense(), output_grad, grd);
                }
            }
            SoftmaxCrossEntropyLoss(a, b) => {
                let a = &mut *get(*a);
                let b = &mut *get(*b);

                let smax = internal.get("softmaxed").unwrap().borrow();

                if let Some(grd) = a.gradients.as_mut() {
                    D::backprop_softmax_crossentropy_loss(&smax, b.values.dense(), output_grad, grd);
                }

                if let Some(grd) = b.gradients.as_mut() {
                    D::backprop_softmax_crossentropy_loss(&smax, a.values.dense(), output_grad, grd);
                }
            }
        }
    }
}

fn setup_ones<D: Device>(device: Arc<D>, internal: &mut HashMap<String, RefCell<DenseMatrix<D>>>, batch_size: usize) {
    if let Some(ones) = internal.get_mut("ones") {
        if ones.borrow().size() < batch_size {
            *ones = RefCell::new(DenseMatrix::ones(device, batch_size));
        }
    } else {
        let ones = RefCell::new(DenseMatrix::ones(device, batch_size));
        internal.insert("ones".to_string(), ones);
    }
}

fn setup_softmax<D: Device>(
    device: Arc<D>,
    internal: &mut HashMap<String, RefCell<DenseMatrix<D>>>,
    single_size: usize,
) {
    if !internal.contains_key("softmaxed") {
        let zeros = RefCell::new(DenseMatrix::zeroed(device.clone(), single_size));
        internal.insert("softmaxed".to_string(), zeros);
        let zeros = RefCell::new(DenseMatrix::zeroed(device, single_size));
        internal.insert("individual_losses".to_string(), zeros);
    }
}
