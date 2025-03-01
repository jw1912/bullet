use crate::backend::{
    shape::Shape,
    tensor::{DenseMatrix, Tensor},
    Device, OperationError,
};

use super::linear_comb::backprop_add_single_scaled;

#[allow(clippy::too_many_arguments)]
pub fn affine<D: Device>(
    a: &DenseMatrix<D>,
    shape_a: Shape,
    b: &DenseMatrix<D>,
    shape_b: Shape,
    c: &DenseMatrix<D>,
    shape_c: Shape,
    ones: &D::BufferF32,
    out: &mut DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    let output_shape = shape_a * shape_b;
    assert_eq!(output_shape.size(), c.single_size());
    assert_eq!(output_shape, shape_c);
    assert!(c.batch_size().is_none());

    matmul(a, shape_a, false, b, shape_b, false, out)?;

    let bs = out.batch_size().unwrap_or(1);
    D::add_assign_single_to_batched_scaled(c.single_size(), bs, ones, 1.0, &c.buf, &mut out.buf)
}

pub fn backprop_affine<D: Device>(
    a: &mut Tensor<D>,
    shape_a: Shape,
    b: &mut Tensor<D>,
    shape_b: Shape,
    c: &mut Tensor<D>,
    ones: &D::BufferF32,
    output_grad: &DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    let output_shape = shape_a * shape_b;
    assert_eq!(output_shape.size(), c.values.single_size());
    assert!(c.values.batch_size().is_none());

    backprop_matmul(
        a.values.dense()?,
        a.gradients.as_mut(),
        shape_a,
        false,
        b.values.dense()?,
        b.gradients.as_mut(),
        shape_b,
        false,
        output_grad,
    )?;

    if let Some(grad) = c.gradients.as_mut() {
        backprop_add_single_scaled(ones, 1.0, c.values.dense()?, grad, output_grad)?;
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
            if x != y {
                return Err(OperationError::MismatchedBatchSizes);
            }

            output.set_batch_size(Some(x))?;
            D::sgemm_batched(x, &input_a.buf, shape_a, trans_a, &input_b.buf, shape_b, trans_b, &mut output.buf, false)
        }
        (None, None) => {
            output.set_batch_size(None)?;
            D::sgemm(&input_a.buf, shape_a, trans_a, &input_b.buf, shape_b, trans_b, &mut output.buf, false)
        }
        (None, Some(x)) => {
            if trans_b {
                return Err(OperationError::UnsupportedOperation("matmul single x batched^T".to_string()));
            }

            let shape_b = Shape::new(shape_b.rows(), x * shape_b.cols());
            output.set_batch_size(Some(x))?;
            D::sgemm(&input_a.buf, shape_a, trans_a, &input_b.buf, shape_b, trans_b, &mut output.buf, false)
        }
        (Some(_), None) => Err(OperationError::UnsupportedOperation("matmul batched x single".to_string())),
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
            if trans_b {
                return Err(OperationError::UnsupportedOperation("backprop matmul single x batched^T".to_string()));
            }

            let shape_b = Shape::new(shape_b.rows(), x * shape_b.cols());
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
        (Some(_), None) => Err(OperationError::UnsupportedOperation("backprop matmul batched x single".to_string())),
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
