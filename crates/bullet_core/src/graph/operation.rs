mod concat;
mod linear_comb;
mod matmul;
mod slice;
mod sparse;

use std::{cell::RefCell, collections::HashMap, sync::Arc};

use crate::{
    device::{Device, DeviceBuffer, OperationError},
    graph::Node,
    shape::Shape,
    tensor::DenseMatrix,
};

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
pub struct GraphBuilderError {
    pub op: Box<Operation>,
    pub ty: GraphBuilderErrorType,
}

impl GraphBuilderError {
    pub fn new(op: &Operation, ty: GraphBuilderErrorType) -> Self {
        Self { op: Box::new(*op), ty }
    }
}

#[derive(Clone, Debug)]
pub enum GraphBuilderErrorType {
    InvalidInputShape(Shape),
    MismatchedInputShapes(Vec<Shape>),
    OutOfBounds(Shape, [usize; 2]),
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
                if a.shape.cols() != 1 {
                    return Err(GraphBuilderError::new(self, InvalidInputShape(a.shape)));
                }

                let out = Shape::new(a.shape.rows() + b.shape.rows(), a.shape.cols());
                ret(a.shape.cols() == b.shape.cols(), out, mismatch(&[a, b]))
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
                ret(is.rows() % min == 0, out, GraphBuilderError::new(self, InvalidInputShape(is)))
            }
            PowerError(a, b, _) => ret(a.shape == b.shape, a.shape, mismatch(&[a, b])),
            ReduceAcrossBatch(node) => {
                let is = node.shape;
                ret(is == Shape::new(1, 1), is, GraphBuilderError::new(self, InvalidInputShape(is)))
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
                ret(valid, out, GraphBuilderError::new(self, OutOfBounds(is, [*start, *end])))
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
    pub(super) fn forward_node(&mut self, output_node: Node) -> Result<(), OperationError<D::DeviceError>> {
        use Operation::*;

        let get = |node: Node| self.nodes[node.idx].borrow();

        let output_tensor = &mut *self.nodes[output_node.idx].borrow_mut();
        let op = if let Some(op) = &output_tensor.operation { op } else { return Ok(()) };
        let internal = &mut output_tensor.internal;
        let output = output_tensor.values.dense_mut();
        let outn = output_tensor.own;

        match op {
            Activate(node, act) => {
                let input = get(*node);
                let input = input.values.dense();
                assert_eq!(outn.shape, node.shape);
                output.set_batch_size(input.batch_size())?;
                D::activate(input.size(), &input.buf, &mut output.buf, *act)
            }
            Affine(wn, inp, bn) => {
                let i = get(*inp);
                let w = get(*wn);
                let w = w.values.dense();

                if let Some(bn) = bn {
                    let bs = i.values.batch_size().unwrap_or(1);
                    setup_ones(w.buf.device(), internal, bs)?;
                    let ones = &internal.get("ones").unwrap().borrow().buf;
                    let b = get(*bn);
                    let b = Some((b.values.dense(), ones, bn.shape));
                    matmul::affine(w, wn.shape, &*i, inp.shape, b, output)
                } else {
                    matmul::affine(w, wn.shape, &*i, inp.shape, None, output)
                }
            }
            AffineDualActivate(wn, sn, nn, bn, act) => {
                assert_eq!(sn.shape, nn.shape);
                sparse::sparse_affine_dual(
                    get(*wn).values.dense(),
                    wn.shape,
                    get(*sn).values.sparse(),
                    get(*nn).values.sparse(),
                    sn.shape,
                    get(*bn).values.dense(),
                    bn.shape,
                    output,
                    *act,
                )
            }
            LinearCombination(alpha, an, beta, bn) => {
                let a = get(*an);
                let a = a.values.dense();
                let bs = a.batch_size().unwrap_or(1);
                setup_ones(a.buf.device(), internal, bs)?;
                let ones = &internal.get("ones").unwrap().borrow().buf;
                linear_comb::linear_comb(ones, *alpha, a, an.shape, *beta, get(*bn).values.dense(), bn.shape, output)
            }
            Gather(input, indices) => {
                let input = get(*input);
                let input = input.values.dense();
                let indices = get(*indices);
                let indices = indices.values.sparse();

                let batch_size = indices.batch_size();
                assert_eq!(input.batch_size(), batch_size);
                assert_eq!(indices.nnz, indices.single_size());
                output.set_batch_size(batch_size)?;

                D::gather(
                    batch_size.unwrap_or(1),
                    input.single_size(),
                    output.single_size(),
                    &input.buf,
                    &indices.buf,
                    &mut output.buf,
                )
            }
            Concat(a, b) => concat::concat(get(*a).values.dense(), a.shape, get(*b).values.dense(), b.shape, output),
            Mask(input, mask) => {
                let input = get(*input);
                let input = input.values.dense();
                let mask = get(*mask);
                let mask = mask.values.sparse();

                let batch_size = mask.batch_size();
                let single_size = mask.single_size();
                assert_eq!(input.batch_size(), batch_size);
                assert_eq!(input.single_size(), single_size);
                assert!(mask.nnz <= single_size);
                assert_eq!(output.single_size(), single_size);
                output.set_batch_size(batch_size)?;

                D::mask(batch_size.unwrap_or(1), single_size, mask.nnz, &input.buf, &mask.buf, &mut output.buf)
            }
            PairwiseMul(node, post_concat) => {
                let input = get(*node);
                let input = &input.values;
                assert_eq!(node.shape.cols(), 1);
                assert_eq!(node.shape.size(), input.single_size());
                assert_eq!(node.shape.size() % 2, 0);
                assert_eq!(node.shape.size() / 2, output.single_size());
                output.set_batch_size(input.batch_size())?;
                D::pairwise(
                    input.single_size(),
                    input.batch_size().unwrap_or(1),
                    &input.dense().buf,
                    &mut output.buf,
                    *post_concat,
                )
            }
            PowerError(a, b, p) => {
                let size = a.shape.size();
                assert_eq!(a.shape, b.shape);

                let a = get(*a);
                let a = a.values.dense();
                let b = get(*b);
                let b = b.values.dense();

                assert_eq!(size, a.single_size());
                assert_eq!(size, b.single_size());
                assert_eq!(size, output.single_size());

                let batch_size = a.batch_size();
                assert_eq!(batch_size, b.batch_size());
                output.set_batch_size(batch_size)?;

                D::abs_power_error(*p, size * batch_size.unwrap_or(1), &a.buf, &b.buf, &mut output.buf)
            }
            ReduceAcrossBatch(node) => {
                let input = get(*node);
                let input = input.values.dense();
                setup_ones(input.buf.device(), internal, input.batch_size().unwrap_or(1))?;
                let ones = internal.get("ones").unwrap().borrow();
                assert_eq!(input.single_size(), node.shape.size());
                D::reduce_add(
                    &ones.buf,
                    input.single_size(),
                    input.batch_size().unwrap_or(1),
                    &input.buf,
                    &mut output.buf,
                )
            }
            Select(input, buckets) => {
                let rows = input.shape.rows();
                let num_buckets = buckets.shape.rows();

                assert_eq!(input.shape.cols(), 1);
                assert_eq!(buckets.shape.cols(), 1);
                assert_eq!(rows % num_buckets, 0, "Cannot divide vector evenly among buckets!");

                let input = get(*input);
                let input = input.values.dense();
                let buckets = get(*buckets);
                let buckets = buckets.values.sparse();
                let batch_size = input.batch_size();
                let output_rows = rows / num_buckets;

                assert_eq!(batch_size, buckets.batch_size());
                assert_eq!(buckets.nnz, 1);
                assert_eq!(rows, input.single_size());
                assert_eq!(num_buckets, buckets.single_size());
                assert_eq!(output_rows, output.single_size());

                output.set_batch_size(batch_size)?;

                D::select(batch_size.unwrap_or(1), rows, output_rows, &input.buf, &buckets.buf, &mut output.buf)
            }
            Slice(input, start, end) => {
                slice::slice_vector_batched(input.shape, get(*input).values.dense(), *start, *end, output)
            }
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => {
                let masks = get(*mask);
                let inputs = get(*input);
                let targets = get(*target);
                let masks = masks.values.sparse();
                let inputs = inputs.values.dense();
                let targets = targets.values.dense();

                assert_eq!(mask.shape, input.shape);
                assert_eq!(input.shape.cols(), 1);
                assert_eq!(mask.shape.size(), masks.single_size());
                assert_eq!(input.shape.size(), inputs.single_size());
                assert!(masks.nnz <= inputs.single_size());
                assert_eq!(target.shape, Shape::new(masks.nnz, 1));
                assert_eq!(masks.batch_size(), inputs.batch_size());
                assert_eq!(masks.batch_size(), targets.batch_size());
                assert_eq!(output.single_size(), 1);

                let batch_size = masks.batch_size().unwrap_or(1);
                let single_size = masks.single_size();
                let nnz = masks.nnz;

                setup_softmax(masks.buf.device(), internal, nnz * batch_size)?;

                let mut smax = internal.get("softmaxed").unwrap().borrow_mut();
                let mut indv = internal.get("individual_losses").unwrap().borrow_mut();

                output.set_batch_size(masks.batch_size())?;
                D::softmax_across_batch_masked(batch_size, single_size, nnz, &masks.buf, &inputs.buf, &mut smax.buf)?;
                D::crossentropy_masked(
                    batch_size,
                    single_size,
                    nnz,
                    &masks.buf,
                    &smax.buf,
                    &targets.buf,
                    &mut indv.buf,
                    &mut output.buf,
                )
            }
            SoftmaxCrossEntropyLoss(an, bn) => {
                let a = get(*an);
                let b = get(*bn);
                let a = a.values.dense();
                let b = b.values.dense();

                assert_eq!(an.shape, bn.shape);
                assert_eq!(an.shape.cols(), 1);
                assert_eq!(an.shape.size(), a.single_size());
                assert_eq!(bn.shape.size(), b.single_size());
                assert_eq!(a.batch_size(), b.batch_size());
                assert_eq!(output.single_size(), 1);

                let batch_size = a.batch_size().unwrap_or(1);
                let single_size = a.single_size();

                setup_softmax(a.buf.device(), internal, single_size * batch_size)?;
                setup_ones(a.buf.device(), internal, single_size)?;

                let ones = internal.get("ones").unwrap().borrow();
                let mut smax = internal.get("softmaxed").unwrap().borrow_mut();
                let mut indv = internal.get("individual_losses").unwrap().borrow_mut();

                D::softmax_across_batch(batch_size, single_size, &a.buf, &mut smax.buf)?;
                D::crossentropy(batch_size * single_size, &smax.buf, &b.buf, &mut indv.buf)?;

                output.set_batch_size(a.batch_size())?;
                D::sgemm(
                    &ones.buf,
                    Shape::new(1, single_size),
                    false,
                    &indv.buf,
                    Shape::new(single_size, batch_size),
                    false,
                    &mut output.buf,
                    false,
                )
            }
        }
    }

    pub(super) fn backward_node(&mut self, output_node: Node) -> Result<(), OperationError<D::DeviceError>> {
        use Operation::*;

        let get = |node: Node| self.nodes[node.idx].borrow_mut();

        let output_tensor = &mut *self.nodes[output_node.idx].borrow_mut();
        let op = if let Some(op) = &output_tensor.operation { op } else { return Ok(()) };
        let internal = &mut output_tensor.internal;
        let outn = output_tensor.own;
        let output_grad = if let Some(grad) = output_tensor.gradients.as_ref() {
            grad
        } else {
            return Ok(());
        };

        match op {
            Activate(node, act) => {
                let input = &mut *get(*node);
                if let Some(grad) = input.gradients.as_mut() {
                    let input = input.values.dense();
                    assert_eq!(outn.shape, node.shape);
                    assert_eq!(output_grad.size(), input.size());
                    assert_eq!(output_grad.batch_size(), input.batch_size());
                    grad.set_batch_size(output_grad.batch_size())?;
                    D::backprop_activate(input.size(), &input.buf, &mut grad.buf, &output_grad.buf, *act)?;
                }
            }
            Affine(wn, inp, bn) => {
                let i = &mut *get(*inp);
                let w = &mut *get(*wn);
                let o = output_tensor.values.dense();

                if let Some(b) = bn {
                    let bs = i.values.batch_size().unwrap_or(1);
                    setup_ones(w.values.dense().buf.device(), internal, bs)?;
                    let ones = &internal.get("ones").unwrap().borrow().buf;
                    matmul::backprop_affine(w, wn.shape, i, inp.shape, Some((&mut *get(*b), ones)), o, output_grad)?;
                } else {
                    matmul::backprop_affine(w, wn.shape, i, inp.shape, None, o, output_grad)?;
                }
            }
            AffineDualActivate(wn, sn, nn, bn, act) => {
                let w = &mut *get(*wn);
                let b = &mut *get(*bn);
                assert_eq!(sn.shape, nn.shape);
                sparse::backprop_sparse_affine_dual_activate(
                    w.values.dense(),
                    w.gradients.as_mut(),
                    wn.shape,
                    get(*sn).values.sparse(),
                    get(*nn).values.sparse(),
                    sn.shape,
                    b.values.dense(),
                    b.gradients.as_mut(),
                    bn.shape,
                    output_tensor.values.dense(),
                    output_grad,
                    *act,
                )?;
            }
            LinearCombination(alpha, an, beta, bn) => {
                let a = &mut *get(*an);
                let b = &mut *get(*bn);

                let abs = a.values.batch_size().unwrap_or(1);
                let bbs = b.values.batch_size().unwrap_or(1);
                let bs = abs.max(bbs);
                setup_ones(a.values.dense().buf.device(), internal, bs)?;
                let ones = &internal.get("ones").unwrap().borrow().buf;

                linear_comb::linear_comb_backward(
                    ones,
                    *alpha,
                    a.values.dense(),
                    a.gradients.as_mut(),
                    *beta,
                    b.values.dense(),
                    b.gradients.as_mut(),
                    output_grad,
                )?;
            }
            Gather(input, indices) => {
                let input = &mut *get(*input);
                let indices = get(*indices);
                let indices = indices.values.sparse();

                if let Some(grd) = input.gradients.as_mut() {
                    let batch_size = indices.batch_size();
                    let input_size = input.values.single_size();
                    assert_eq!(batch_size, input.values.batch_size());
                    assert_eq!(batch_size, output_grad.batch_size());
                    assert_eq!(indices.nnz, indices.single_size());
                    assert_eq!(indices.nnz, output_grad.single_size());

                    grd.set_batch_size(batch_size)?;
                    D::backprop_gather(
                        batch_size.unwrap_or(1),
                        input_size,
                        output_grad.single_size(),
                        &output_grad.buf,
                        &indices.buf,
                        &mut grd.buf,
                    )?;
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
                )?;
            }
            Mask(input, mask) => {
                if let Some(grd) = get(*input).gradients.as_mut() {
                    let mask = get(*mask);
                    let mask = mask.values.sparse();
                    let batch_size = mask.batch_size();
                    let single_size = mask.single_size();

                    grd.set_batch_size(batch_size)?;
                    D::backprop_mask(
                        batch_size.unwrap_or(1),
                        single_size,
                        mask.nnz,
                        &output_grad.buf,
                        &mask.buf,
                        &mut grd.buf,
                    )?;
                }
            }
            PairwiseMul(node, post_concat) => {
                let input = &mut *get(*node);
                if let Some(grd) = input.gradients.as_mut() {
                    let input = &input.values;
                    assert_eq!(node.shape.size() % 2, 0);
                    assert_eq!(node.shape.size(), input.single_size());
                    assert_eq!(node.shape.size() / 2, output_grad.single_size());
                    assert_eq!(node.shape.size(), grd.single_size());
                    assert_eq!(input.batch_size(), output_grad.batch_size());
                    grd.set_batch_size(input.batch_size())?;
                    D::backprop_pairwise(
                        input.single_size(),
                        input.batch_size().unwrap_or(1),
                        &input.dense().buf,
                        &output_grad.buf,
                        &mut grd.buf,
                        *post_concat,
                    )?;
                }
            }
            PowerError(a, b, p) => {
                let size = a.shape.size();
                assert_eq!(a.shape, b.shape);

                let a = &mut *get(*a);
                let b = &mut *get(*b);

                assert_eq!(size, a.values.single_size());
                assert_eq!(size, b.values.single_size());
                assert_eq!(size, output_grad.single_size());

                let batch_size = a.values.batch_size();
                assert_eq!(batch_size, b.values.batch_size());
                assert_eq!(batch_size, output_grad.batch_size());

                if let Some(grd) = a.gradients.as_mut() {
                    assert_eq!(size, grd.single_size());
                    grd.set_batch_size(batch_size)?;
                    D::backprop_abs_power_error_single(
                        *p,
                        size * batch_size.unwrap_or(1),
                        &a.values.dense().buf,
                        &b.values.dense().buf,
                        &output_grad.buf,
                        &mut grd.buf,
                    )?;
                }

                if let Some(grd) = b.gradients.as_mut() {
                    assert_eq!(size, grd.single_size());
                    grd.set_batch_size(batch_size)?;
                    D::backprop_abs_power_error_single(
                        *p,
                        size * batch_size.unwrap_or(1),
                        &b.values.dense().buf,
                        &a.values.dense().buf,
                        &output_grad.buf,
                        &mut grd.buf,
                    )?;
                }
            }
            ReduceAcrossBatch(input) => {
                let input = &mut *get(*input);
                if let Some(grd) = input.gradients.as_mut() {
                    let vals = input.values.dense();
                    let bs = vals.batch_size();
                    let ss = vals.single_size();

                    setup_ones(vals.buf.device(), internal, bs.unwrap_or(1))?;
                    let ones = &internal.get("ones").unwrap().borrow().buf;

                    assert!(output_grad.batch_size().is_none());
                    assert_eq!(vals.single_size(), output_grad.single_size());
                    assert_eq!(vals.single_size(), grd.single_size());

                    grd.set_batch_size(bs)?;
                    D::add_assign_single_to_batched_scaled(
                        ss,
                        bs.unwrap_or(1),
                        ones,
                        1.0,
                        &output_grad.buf,
                        &mut grd.buf,
                    )?;
                }
            }
            Select(input, buckets) => {
                let rows = input.shape.rows();
                let num_buckets = buckets.shape.rows();

                assert_eq!(input.shape.cols(), 1);
                assert_eq!(buckets.shape.cols(), 1);
                assert_eq!(rows % num_buckets, 0, "Cannot divide vector evenly among buckets!");

                let input = &mut *get(*input);

                if let Some(grd) = input.gradients.as_mut() {
                    let input = input.values.dense();
                    let buckets = get(*buckets);
                    let buckets = buckets.values.sparse();
                    let batch_size = input.batch_size();
                    let output_rows = rows / num_buckets;

                    assert_eq!(rows, input.single_size());
                    assert_eq!(num_buckets, buckets.single_size());
                    assert_eq!(batch_size, buckets.batch_size());
                    assert_eq!(batch_size, output_grad.batch_size());
                    assert_eq!(buckets.nnz, 1);
                    assert_eq!(output_rows, output_grad.single_size());

                    grd.set_batch_size(batch_size)?;

                    D::select_backprop(
                        batch_size.unwrap_or(1),
                        rows,
                        output_rows,
                        &buckets.buf,
                        &output_grad.buf,
                        &mut grd.buf,
                    )?;
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
                    )?;
                }
            }
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => {
                let masks = &*get(*mask);
                let masks = masks.values.sparse();
                let inputs = &mut *get(*input);
                let targets = &mut *get(*target);
                let targets = targets.values.dense();

                let smax = internal.get("softmaxed").unwrap().borrow();
                let batch_size = masks.batch_size();
                let single_size = masks.single_size();
                let nnz = masks.nnz;

                assert_eq!(batch_size, inputs.values.batch_size());
                assert_eq!(batch_size, targets.batch_size());
                assert_eq!(batch_size, output_grad.batch_size());

                if let Some(grd) = inputs.gradients.as_mut() {
                    grd.set_batch_size(batch_size)?;
                    D::backprop_softmax_crossentropy_masked(
                        batch_size.unwrap_or(1),
                        single_size,
                        nnz,
                        &masks.buf,
                        &smax.buf,
                        &targets.buf,
                        &output_grad.buf,
                        &mut grd.buf,
                    )?;
                }
            }
            SoftmaxCrossEntropyLoss(an, bn) => {
                let a = &mut *get(*an);
                let b = &mut *get(*bn);

                assert_eq!(an.shape, bn.shape);
                assert_eq!(an.shape.cols(), 1);
                assert_eq!(an.shape.size(), a.values.single_size());
                assert_eq!(bn.shape.size(), b.values.single_size());
                assert_eq!(a.values.batch_size(), b.values.batch_size());
                assert_eq!(a.values.batch_size(), output_grad.batch_size());
                assert_eq!(output_grad.single_size(), 1);

                let ones = internal.get("ones").unwrap().borrow();
                let smax = internal.get("softmaxed").unwrap().borrow();
                let mut indv = internal.get("individual_losses").unwrap().borrow_mut();

                let batch_size = a.values.batch_size();
                let single_size = a.values.single_size();
                let size = single_size * batch_size.unwrap_or(1);

                D::sgemm(
                    &ones.buf,
                    Shape::new(single_size, 1),
                    false,
                    &output_grad.buf,
                    Shape::new(1, batch_size.unwrap_or(1)),
                    false,
                    &mut indv.buf,
                    false,
                )?;

                let smax = &smax.buf;
                let indv = &indv.buf;

                if let Some(grd) = a.gradients.as_mut() {
                    grd.set_batch_size(batch_size)?;
                    D::backprop_softmax_crossentropy(size, smax, &b.values.dense().buf, indv, &mut grd.buf)?;
                }

                if let Some(grd) = b.gradients.as_mut() {
                    grd.set_batch_size(batch_size)?;
                    D::backprop_softmax_crossentropy(size, smax, &a.values.dense().buf, indv, &mut grd.buf)?;
                }
            }
        }

        Ok(())
    }
}

fn setup_ones<D: Device>(
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

fn setup_softmax<D: Device>(
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
