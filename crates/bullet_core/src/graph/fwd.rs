use crate::{
    backend::{activation::Activation, error::OperationError, shape::Shape, Device, DeviceBuffer},
    graph::{Graph, Node},
};

use super::operation::{concat, linear_comb, matmul, setup_ones, setup_softmax, slice, sparse, Operation};

impl<D: Device> Graph<D> {
    pub(crate) fn forward_node(&mut self, output_node: Node) -> Result<(), OperationError<D::DeviceError>> {
        use Operation::*;

        let get = |node: Node| self.get(node.idx).unwrap();

        let output_tensor = &mut *self.get_mut(output_node.idx)?;
        let op = if let Some(op) = &output_tensor.operation { op } else { return Ok(()) };
        let internal = &mut output_tensor.internal;
        let output = output_tensor.values.dense_mut()?;
        let outn = output_tensor.own;

        match op {
            Activate(node, act) => {
                let input = get(*node);
                let input = input.values.dense()?;
                assert_eq!(outn.shape, node.shape);
                output.set_batch_size(input.batch_size())?;
                D::activate(input.size(), &input.buf, &mut output.buf, *act)
            }
            Affine(wn, inp, bn) => {
                let w = get(*wn);
                let i = get(*inp);
                let b = get(*bn);
                let w = w.values.dense()?;
                let i = i.values.dense()?;
                let b = b.values.dense()?;

                let bs = i.batch_size().unwrap_or(1);
                setup_ones(w.buf.device(), internal, bs)?;
                let ones = &internal.get("ones").unwrap().borrow().buf;
                matmul::affine(w, wn.shape, i, inp.shape, b, bn.shape, ones, output)
            }
            LinearCombination(alpha, an, beta, bn) => {
                let a = get(*an);
                let a = a.values.dense()?;
                let bs = a.batch_size().unwrap_or(1);
                setup_ones(a.buf.device(), internal, bs)?;
                let ones = &internal.get("ones").unwrap().borrow().buf;
                linear_comb::linear_comb(ones, *alpha, a, an.shape, *beta, get(*bn).values.dense()?, bn.shape, output)
            }
            Gather(input, indices) => {
                let input = get(*input);
                let input = input.values.dense()?;
                let indices = get(*indices);
                let indices = indices.values.sparse()?;

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
            Concat(a, b) => concat::concat(get(*a).values.dense()?, a.shape, get(*b).values.dense()?, b.shape, output),
            Mask(input, mask) => {
                let input = get(*input);
                let input = input.values.dense()?;
                let mask = get(*mask);
                let mask = mask.values.sparse()?;

                let batch_size = mask.batch_size();
                let single_size = mask.single_size();
                assert_eq!(input.batch_size(), batch_size);
                assert_eq!(input.single_size(), single_size);
                assert!(mask.nnz <= single_size);
                assert_eq!(output.single_size(), single_size);
                output.set_batch_size(batch_size)?;

                D::mask(batch_size.unwrap_or(1), single_size, mask.nnz, &input.buf, &mask.buf, &mut output.buf)
            }
            Matmul(a, trans_a, b, trans_b) => matmul::matmul(
                get(*a).values.dense()?,
                a.shape,
                *trans_a,
                get(*b).values.dense()?,
                b.shape,
                *trans_b,
                output,
            ),
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
                    &input.dense()?.buf,
                    &mut output.buf,
                    *post_concat,
                )
            }
            PowerError(a, b, p) => {
                let size = a.shape.size();
                assert_eq!(a.shape, b.shape);

                let a = get(*a);
                let a = a.values.dense()?;
                let b = get(*b);
                let b = b.values.dense()?;

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
                let input = input.values.dense()?;
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
                let input = input.values.dense()?;
                let buckets = get(*buckets);
                let buckets = buckets.values.sparse()?;
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
                slice::slice_vector_batched(input.shape, get(*input).values.dense()?, *start, *end, output)
            }
            SparseAffine(wn, inp, bn) => {
                let i = get(*inp);
                let w = get(*wn);
                let w = w.values.dense()?;

                if let Some(bn) = bn {
                    let b = get(*bn);
                    let b = Some((b.values.dense()?, bn.shape));
                    sparse::affine_activate(
                        None,
                        Activation::Identity,
                        w,
                        wn.shape,
                        i.values.sparse()?,
                        inp.shape,
                        b,
                        output,
                    )
                } else {
                    sparse::affine_activate(
                        None,
                        Activation::Identity,
                        w,
                        wn.shape,
                        i.values.sparse()?,
                        inp.shape,
                        None,
                        output,
                    )
                }
            }
            SparseAffineDualActivate(wn, sn, nn, bn, act) => {
                assert_eq!(sn.shape, nn.shape);
                sparse::affine_dual(
                    get(*wn).values.dense()?,
                    wn.shape,
                    get(*sn).values.sparse()?,
                    get(*nn).values.sparse()?,
                    sn.shape,
                    get(*bn).values.dense()?,
                    bn.shape,
                    output,
                    *act,
                )
            }
            ToDense(node) => get(*node).values.sparse()?.copy_into_dense(output),
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => {
                let masks = get(*mask);
                let inputs = get(*input);
                let targets = get(*target);
                let masks = masks.values.sparse()?;
                let inputs = inputs.values.dense()?;
                let targets = targets.values.dense()?;

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
                let a = a.values.dense()?;
                let b = b.values.dense()?;

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
}
