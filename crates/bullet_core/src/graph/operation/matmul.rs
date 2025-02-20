use crate::{
    device::{Device, OperationError},
    shape::Shape,
    tensor::{DenseMatrix, Matrix, Tensor},
};

use super::linear_comb::backprop_add_single_scaled;

pub fn affine<D: Device>(
    a: &DenseMatrix<D>,
    shape_a: Shape,
    b: &Tensor<D>,
    shape_b: Shape,
    c: Option<(&DenseMatrix<D>, &D::BufferF32, Shape)>,
    out: &mut DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    let output_shape = shape_a * shape_b;
    if let Some((c, _, shape)) = c {
        assert_eq!(output_shape.size(), c.single_size());
        assert_eq!(output_shape, shape);
        assert!(c.batch_size().is_none());
    }

    match &b.values {
        Matrix::Dense(dense) => {
            matmul(a, shape_a, false, dense, shape_b, false, out)?;

            if let Some((c, ones, _)) = c {
                let bs = out.batch_size().unwrap_or(1);
                D::add_assign_single_to_batched_scaled(c.single_size(), bs, ones, 1.0, &c.buf, &mut out.buf)?;
            }
        }
        Matrix::Sparse(sparse) => {
            assert_eq!(shape_b.cols(), 1);
            assert_eq!(a.single_size(), shape_a.size());
            assert_eq!(sparse.single_size(), shape_b.size());
            assert!(a.batch_size().is_none());

            let bs = sparse.batch_size();
            out.set_batch_size(bs)?;

            D::sparse_affine(
                bs.unwrap_or(1),
                &a.buf,
                shape_a,
                &sparse.buf,
                shape_b,
                sparse.nnz,
                c.map(|x| &x.0.buf),
                &mut out.buf,
            )?;
        }
    }

    Ok(())
}

pub fn backprop_affine<D: Device>(
    a: &mut Tensor<D>,
    shape_a: Shape,
    b: &mut Tensor<D>,
    shape_b: Shape,
    c: Option<(&mut Tensor<D>, &D::BufferF32)>,
    output: &DenseMatrix<D>,
    output_grad: &DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    let output_shape = shape_a * shape_b;
    if let Some((c, _)) = &c {
        assert_eq!(output_shape.size(), c.values.single_size());
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
            )?;

            if let Some((c, ones)) = c {
                if let Some(grad) = c.gradients.as_mut() {
                    backprop_add_single_scaled(ones, 1.0, c.values.dense(), grad, output_grad)?;
                }
            }
        }
        Matrix::Sparse(sparse) => {
            assert_eq!(shape_b.cols(), 1);
            assert!(b.gradients.is_none());
            assert_eq!(a.values.single_size(), shape_a.size());
            assert_eq!(sparse.single_size(), shape_b.size());
            assert!(a.values.batch_size().is_none());
            assert_eq!(b.values.batch_size(), output.batch_size());
            assert_eq!(b.values.batch_size(), output_grad.batch_size());

            if let Some((c, _)) = &c {
                assert!(c.values.batch_size().is_none());
                assert_eq!(c.values.single_size(), output_shape.size());
            }

            if let Some(agrd) = a.gradients.as_mut() {
                let (c, cgrd) = c
                    .map(|(c, _)| (Some(&c.values.dense().buf), c.gradients.as_mut().map(|g| &mut g.buf)))
                    .unwrap_or_default();

                D::backprop_sparse_affine(
                    output_grad.batch_size().unwrap_or(1),
                    &a.values.dense().buf,
                    &mut agrd.buf,
                    shape_a,
                    &sparse.buf,
                    shape_b,
                    sparse.nnz,
                    c,
                    cgrd,
                    &output.buf,
                    &output_grad.buf,
                )?;
            } else if let Some((c, ones)) = c {
                if let Some(grad) = c.gradients.as_mut() {
                    backprop_add_single_scaled(ones, 1.0, c.values.dense(), grad, output_grad)?;
                }
            }
        }
    }

    Ok(())
}

pub fn matmul<D: Device>(
    input_a: &DenseMatrix<D>,
    shape_a: Shape,
    trans_a: bool,
    input_b: &DenseMatrix<D>,
    shape_b: Shape,
    trans_b: bool,
    output: &mut DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    let output_shape = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);
    assert_eq!(output_shape.size(), output.single_size());

    match (input_a.batch_size(), input_b.batch_size()) {
        (Some(x), Some(y)) => {
            assert_eq!(x, y);
            output.set_batch_size(Some(x))?;
            D::sgemm_batched(x, &input_a.buf, shape_a, trans_a, &input_b.buf, shape_b, trans_b, &mut output.buf, false)
        }
        (None, None) => {
            output.set_batch_size(None)?;
            D::sgemm(&input_a.buf, shape_a, trans_a, &input_b.buf, shape_b, trans_b, &mut output.buf, false)
        }
        (None, Some(x)) => {
            if trans_b || shape_b.cols() > 1 {
                println!("{trans_b}, {shape_b}");
                return Err(OperationError::UnsupportedOperation);
            }

            let shape_b = Shape::new(shape_b.rows(), x);
            output.set_batch_size(Some(x))?;
            D::sgemm(&input_a.buf, shape_a, trans_a, &input_b.buf, shape_b, trans_b, &mut output.buf, false)
        }
        (Some(_), None) => Err(OperationError::UnsupportedOperation),
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
) -> Result<(), OperationError<D::DeviceError>> {
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
            )
        }
        (None, None) => backprop_single_matmul(
            input_a,
            shape_a,
            input_a_grad,
            trans_a,
            input_b,
            shape_b,
            input_b_grad,
            trans_b,
            output_grad,
        ),
        (None, Some(x)) => {
            if trans_b || shape_b.cols() > 1 {
                unimplemented!()
            }

            let shape_b = Shape::new(shape_b.rows(), x);
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
            )
        }
        (Some(_), None) => Err(OperationError::UnsupportedOperation),
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
) -> Result<(), OperationError<D::DeviceError>> {
    let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);

    if let Some(grad_a) = input_a_grad {
        grad_a.set_batch_size(input_a.batch_size())?;
        if trans_a {
            D::sgemm(&input_b.buf, shape_b, trans_b, &output_grad.buf, shape_o, true, &mut grad_a.buf, true)?;
        } else {
            D::sgemm(&output_grad.buf, shape_o, false, &input_b.buf, shape_b, !trans_b, &mut grad_a.buf, true)?;
        }
    }

    if let Some(grad_b) = input_b_grad {
        grad_b.set_batch_size(input_b.batch_size())?;
        if trans_b {
            D::sgemm(&output_grad.buf, shape_o, true, &input_a.buf, shape_a, trans_a, &mut grad_b.buf, true)?;
        } else {
            D::sgemm(&input_a.buf, shape_a, !trans_a, &output_grad.buf, shape_o, false, &mut grad_b.buf, true)?;
        }
    }

    Ok(())
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
) -> Result<(), OperationError<D::DeviceError>> {
    let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);
    assert_eq!(output_grad.single_size(), shape_o.size());
    assert_eq!(input_a.batch_size(), input_b.batch_size());
    assert_eq!(input_a.batch_size(), output_grad.batch_size());

    let bs = input_a.batch_size().unwrap_or(1);

    if let Some(grad_a) = input_a_grad {
        grad_a.set_batch_size(input_a.batch_size())?;
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
            )?;
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
            )?;
        }
    }

    if let Some(grad_b) = input_b_grad {
        grad_b.set_batch_size(input_b.batch_size())?;
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
            )?;
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
            )?;
        }
    }

    Ok(())
}
