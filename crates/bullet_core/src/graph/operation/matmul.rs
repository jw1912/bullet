use crate::{
    device::Device,
    shape::Shape,
    tensor::{DenseMatrix, Matrix, Tensor},
};

pub fn affine<D: Device>(
    a: &DenseMatrix<D>,
    shape_a: Shape,
    b: &Tensor<D>,
    shape_b: Shape,
    c: Option<(&DenseMatrix<D>, &D::BufferF32)>,
    out: &mut DenseMatrix<D>,
) {
    let output_size = shape_a * shape_b;
    if let Some((c, _)) = c {
        assert_eq!(output_size.size(), c.single_size());
        assert!(c.batch_size().is_none());
    }

    match &b.values {
        Matrix::Dense(dense) => {
            matmul(a, shape_a, false, dense, shape_b, false, out);
            if let Some((c, ones)) = c {
                let bs = out.batch_size().unwrap_or(1);
                D::add_assign_single_to_batched_scaled(c.single_size(), bs, ones, 1.0, &c.buf, &mut out.buf);
            }
        }
        Matrix::Sparse(sparse) => D::sparse_affine(a, sparse, c.map(|x| x.0), out),
    }
}

pub fn backprop_affine<D: Device>(
    a: &mut Tensor<D>,
    shape_a: Shape,
    b: &mut Tensor<D>,
    shape_b: Shape,
    c: Option<(&mut Tensor<D>, &D::BufferF32)>,
    output: &DenseMatrix<D>,
    output_grad: &DenseMatrix<D>,
) {
    let output_size = shape_a * shape_b;
    if let Some((c, _)) = &c {
        assert_eq!(output_size.size(), c.values.single_size());
        assert!(c.values.batch_size().is_none());
    }

    match &b.values {
        Matrix::Dense(dense) => {
            backprop_matmul(
                a.values.dense(),
                a.gradients.as_mut(),
                shape_a,
                false,
                dense,
                b.gradients.as_mut(),
                shape_b,
                false,
                output_grad,
            );

            if let Some((c, ones)) = c {
                if let Some(grad) = c.gradients.as_mut() {
                    D::backprop_add_single_scaled(ones, 1.0, c.values.dense(), grad, output_grad);
                }
            }
        }
        Matrix::Sparse(sparse) => {
            assert!(b.gradients.is_none());

            if let Some(agrd) = a.gradients.as_mut() {
                let (c, cgrd) = c.map(|x| (Some(x.0.values.dense()), x.0.gradients.as_mut())).unwrap_or((None, None));
                D::backprop_sparse_affine(a.values.dense(), agrd, sparse, c, cgrd, output, output_grad)
            } else if let Some((c, ones)) = c {
                if let Some(grad) = c.gradients.as_mut() {
                    D::backprop_add_single_scaled(ones, 1.0, c.values.dense(), grad, output_grad);
                }
            }
        }
    }
}

pub fn matmul<D: Device>(
    input_a: &DenseMatrix<D>,
    shape_a: Shape,
    trans_a: bool,
    input_b: &DenseMatrix<D>,
    shape_b: Shape,
    trans_b: bool,
    output: &mut DenseMatrix<D>,
) {
    let output_shape = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);
    assert_eq!(output_shape.size(), output.single_size());

    match (input_a.batch_size(), input_b.batch_size()) {
        (Some(x), Some(y)) => {
            assert_eq!(x, y);
            output.set_batch_size(Some(x));
            D::sgemm_batched(x, &input_a.buf, shape_a, trans_a, &input_b.buf, shape_b, trans_b, &mut output.buf, false)
        }
        (None, None) => {
            output.set_batch_size(None);
            D::sgemm(&input_a.buf, shape_a, trans_a, &input_b.buf, shape_b, trans_b, &mut output.buf, false);
        }
        (None, Some(x)) => {
            let shape_b = Shape::new(shape_b.rows(), x);
            if trans_b || shape_b.cols() > 1 {
                unimplemented!()
            }

            output.set_batch_size(Some(x));
            D::sgemm(&input_a.buf, shape_a, trans_a, &input_b.buf, shape_b, trans_b, &mut output.buf, false);
        }
        (Some(_), None) => unimplemented!(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn backprop_matmul<D: Device>(
    input_a: &DenseMatrix<D>,
    input_a_grad: Option<&mut DenseMatrix<D>>,
    shape_a: Shape,
    trans_a: bool,
    input_b: &DenseMatrix<D>,
    input_b_grad: Option<&mut DenseMatrix<D>>,
    shape_b: Shape,
    trans_b: bool,
    output_grad: &DenseMatrix<D>,
) {
    match (input_a.batch_size(), input_b.batch_size()) {
        (Some(x), Some(y)) => {
            assert_eq!(x, y);
            backprop_batched_matmul(
                input_a,
                input_a_grad,
                shape_a,
                trans_a,
                input_b,
                input_b_grad,
                shape_b,
                trans_b,
                output_grad,
            );
        }
        (None, None) => {
            backprop_single_matmul(
                input_a,
                shape_a,
                input_a_grad,
                trans_a,
                input_b,
                shape_b,
                input_b_grad,
                trans_b,
                output_grad,
            );
        }
        (None, Some(x)) => {
            let shape_b = Shape::new(shape_b.rows(), x);
            if trans_b || shape_b.cols() > 1 {
                unimplemented!()
            }

            backprop_single_matmul(
                input_a,
                shape_a,
                input_a_grad,
                trans_a,
                input_b,
                shape_b,
                input_b_grad,
                trans_b,
                output_grad,
            );
        }
        (Some(_), None) => unimplemented!(),
    }
}

#[allow(clippy::too_many_arguments)]
fn backprop_single_matmul<D: Device>(
    input_a: &DenseMatrix<D>,
    shape_a: Shape,
    input_a_grad: Option<&mut DenseMatrix<D>>,
    trans_a: bool,
    input_b: &DenseMatrix<D>,
    shape_b: Shape,
    input_b_grad: Option<&mut DenseMatrix<D>>,
    trans_b: bool,
    output_grad: &DenseMatrix<D>,
) {
    let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);

    if let Some(grad_a) = input_a_grad {
        grad_a.set_batch_size(input_a.batch_size());
        if trans_a {
            D::sgemm(&input_b.buf, shape_b, trans_b, &output_grad.buf, shape_o, true, &mut grad_a.buf, true);
        } else {
            D::sgemm(&output_grad.buf, shape_o, false, &input_b.buf, shape_b, !trans_b, &mut grad_a.buf, true);
        }
    }

    if let Some(grad_b) = input_b_grad {
        grad_b.set_batch_size(input_b.batch_size());
        if trans_b {
            D::sgemm(&output_grad.buf, shape_o, true, &input_a.buf, shape_a, trans_a, &mut grad_b.buf, true);
        } else {
            D::sgemm(&input_a.buf, shape_a, !trans_a, &output_grad.buf, shape_o, false, &mut grad_b.buf, true);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn backprop_batched_matmul<D: Device>(
    input_a: &DenseMatrix<D>,
    input_a_grad: Option<&mut DenseMatrix<D>>,
    shape_a: Shape,
    trans_a: bool,
    input_b: &DenseMatrix<D>,
    input_b_grad: Option<&mut DenseMatrix<D>>,
    shape_b: Shape,
    trans_b: bool,
    output_grad: &DenseMatrix<D>,
) {
    let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);
    assert_eq!(output_grad.single_size(), shape_o.size());
    assert_eq!(input_a.batch_size(), input_b.batch_size());
    assert_eq!(input_a.batch_size(), output_grad.batch_size());

    let bs = input_a.batch_size().unwrap_or(1);

    if let Some(grad_a) = input_a_grad {
        grad_a.set_batch_size(input_a.batch_size());
        if trans_a {
            D::sgemm_batched(
                bs,
                &input_b.buf,
                shape_b,
                trans_b,
                &output_grad.buf,
                shape_o,
                true,
                &mut grad_a.buf,
                true,
            );
        } else {
            D::sgemm_batched(
                bs,
                &output_grad.buf,
                shape_o,
                false,
                &input_b.buf,
                shape_b,
                !trans_b,
                &mut grad_a.buf,
                true,
            );
        }
    }

    if let Some(grad_b) = input_b_grad {
        grad_b.set_batch_size(input_b.batch_size());
        if trans_b {
            D::sgemm_batched(
                bs,
                &output_grad.buf,
                shape_o,
                true,
                &input_a.buf,
                shape_a,
                trans_a,
                &mut grad_b.buf,
                true,
            );
        } else {
            D::sgemm_batched(
                bs,
                &input_a.buf,
                shape_a,
                !trans_a,
                &output_grad.buf,
                shape_o,
                false,
                &mut grad_b.buf,
                true,
            );
        }
    }
}
