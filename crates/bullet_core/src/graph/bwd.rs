use crate::{
    backend::{activation::Activation, error::OperationError, shape::Shape, Device, DeviceBuffer},
    graph::{Graph, Node},
};

use super::operation::{concat, linear_comb, matmul, setup_ones, slice, sparse, Operation};

impl<D: Device> Graph<D> {
    pub(crate) fn backward_node(&mut self, output_node: Node) -> Result<(), OperationError<D::DeviceError>> {
        use Operation::*;

        let get = |node: Node| self.get_mut(node.idx).unwrap();

        let output_tensor = &mut *self.get_mut(output_node.idx)?;
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
                    let input = input.values.dense()?;
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
                let bs = i.values.batch_size().unwrap_or(1);
                setup_ones(w.values.dense()?.buf.device(), internal, bs)?;
                let ones = &internal.get("ones").unwrap().borrow().buf;
                matmul::backprop_affine(w, wn.shape, i, inp.shape, &mut *get(*bn), ones, output_grad)?;
            }
            LinearCombination(alpha, an, beta, bn) => {
                let a = &mut *get(*an);
                let b = &mut *get(*bn);

                let abs = a.values.batch_size().unwrap_or(1);
                let bbs = b.values.batch_size().unwrap_or(1);
                let bs = abs.max(bbs);
                setup_ones(a.values.dense()?.buf.device(), internal, bs)?;
                let ones = &internal.get("ones").unwrap().borrow().buf;

                linear_comb::linear_comb_backward(
                    ones,
                    *alpha,
                    a.values.dense()?,
                    a.gradients.as_mut(),
                    *beta,
                    b.values.dense()?,
                    b.gradients.as_mut(),
                    output_grad,
                )?;
            }
            Gather(input, indices) => {
                let input = &mut *get(*input);
                let indices = get(*indices);
                let indices = indices.values.sparse()?;

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
                    a.values.dense()?,
                    a.gradients.as_mut(),
                    an.shape,
                    b.values.dense()?,
                    b.gradients.as_mut(),
                    bn.shape,
                    output_grad,
                )?;
            }
            Mask(input, mask) => {
                if let Some(grd) = get(*input).gradients.as_mut() {
                    let mask = get(*mask);
                    let mask = mask.values.sparse()?;
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
            Matmul(an, trans_a, bn, trans_b) => {
                let a = &mut *get(*an);
                let b = &mut *get(*bn);

                matmul::backprop_matmul(
                    a.values.dense()?,
                    a.gradients.as_mut(),
                    an.shape,
                    *trans_a,
                    b.values.dense()?,
                    b.gradients.as_mut(),
                    bn.shape,
                    *trans_b,
                    output_grad,
                )?;
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
                        &input.dense()?.buf,
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
                        &a.values.dense()?.buf,
                        &b.values.dense()?.buf,
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
                        &b.values.dense()?.buf,
                        &a.values.dense()?.buf,
                        &output_grad.buf,
                        &mut grd.buf,
                    )?;
                }
            }
            ReduceAcrossBatch(input) => {
                let input = &mut *get(*input);
                if let Some(grd) = input.gradients.as_mut() {
                    let vals = input.values.dense()?;
                    let bs = vals.batch_size();
                    let ss = vals.single_size();

                    setup_ones(vals.buf.device(), internal, bs.unwrap_or(1))?;
                    let ones = &internal.get("ones").unwrap().borrow().buf;

                    assert!(output_grad.batch_size().is_none());
                    assert_eq!(vals.single_size(), output_grad.single_size());
                    assert_eq!(vals.single_size(), grd.single_size());

                    grd.set_batch_size(bs)?;
                    linear_comb::add_assign_single_to_batched_scaled::<D>(
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
                    let input = input.values.dense()?;
                    let buckets = get(*buckets);
                    let buckets = buckets.values.sparse()?;
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
                        input.values.dense()?,
                        grd,
                        *start,
                        *end,
                        output_grad,
                    )?;
                }
            }
            SparseAffine(wn, inp, bn) => {
                let i = &mut *get(*inp);
                let w = &mut *get(*wn);
                let o = output_tensor.values.dense()?;

                let i = i.values.sparse()?;

                if let Some(b) = bn {
                    let bs = i.batch_size().unwrap_or(1);
                    setup_ones(w.values.dense()?.buf.device(), internal, bs)?;
                    let ones = &internal.get("ones").unwrap().borrow().buf;
                    sparse::backprop_affine_activate(
                        None,
                        Activation::Identity,
                        w,
                        wn.shape,
                        i,
                        inp.shape,
                        &mut Some((&mut *get(*b), ones)),
                        o,
                        output_grad,
                    )?;
                } else {
                    sparse::backprop_affine_activate(
                        None,
                        Activation::Identity,
                        w,
                        wn.shape,
                        i,
                        inp.shape,
                        &mut None,
                        o,
                        output_grad,
                    )?;
                }
            }
            SparseAffineDualActivate(wn, sn, nn, bn, act) => {
                let w = &mut *get(*wn);
                let b = &mut *get(*bn);
                let s = get(*sn);

                let bs = s.values.batch_size().unwrap_or(1);
                assert_eq!(sn.shape, nn.shape);
                setup_ones(w.values.dense()?.buf.device(), internal, bs)?;
                let ones = &internal.get("ones").unwrap().borrow().buf;
                sparse::backprop_affine_dual(
                    w,
                    wn.shape,
                    s.values.sparse()?,
                    get(*nn).values.sparse()?,
                    sn.shape,
                    &mut Some((b, ones)),
                    output_tensor.values.dense()?,
                    output_grad,
                    *act,
                )?;
            }
            ToDense(_) => return Err(OperationError::UnsupportedOperation),
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => {
                let masks = &*get(*mask);
                let masks = masks.values.sparse()?;
                let inputs = &mut *get(*input);
                let targets = &mut *get(*target);
                let targets = targets.values.dense()?;

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
                    1.0,
                    &ones.buf,
                    Shape::new(single_size, 1),
                    false,
                    &output_grad.buf,
                    Shape::new(1, batch_size.unwrap_or(1)),
                    false,
                    0.0,
                    &mut indv.buf,
                )?;

                let smax = &smax.buf;
                let indv = &indv.buf;

                if let Some(grd) = a.gradients.as_mut() {
                    grd.set_batch_size(batch_size)?;
                    D::backprop_softmax_crossentropy(size, smax, &b.values.dense()?.buf, indv, &mut grd.buf)?;
                }

                if let Some(grd) = b.gradients.as_mut() {
                    grd.set_batch_size(batch_size)?;
                    D::backprop_softmax_crossentropy(size, smax, &a.values.dense()?.buf, indv, &mut grd.buf)?;
                }
            }
        }

        Ok(())
    }
}
