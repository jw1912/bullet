use crate::{
    device::{Device, OperationError},
    graph::operation::linear_comb::backprop_add_single_scaled,
    shape::Shape,
    tensor::{DenseMatrix, SparseMatrix, Tensor},
};

use super::Activation;

#[allow(clippy::too_many_arguments)]
pub fn affine_activate<D: Device>(
    stride: Option<bool>,
    activation: Activation,
    a: &DenseMatrix<D>,
    shape_a: Shape,
    b: &SparseMatrix<D>,
    shape_b: Shape,
    c: Option<(&DenseMatrix<D>, Shape)>,
    out: &mut DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    let output_shape = shape_a * shape_b;
    if let Some((c, shape)) = c {
        assert_eq!(output_shape.size(), c.single_size());
        assert_eq!(output_shape, shape);
        assert!(c.batch_size().is_none());
    }

    assert_eq!(shape_b.cols(), 1);
    assert_eq!(a.single_size(), shape_a.size());
    assert_eq!(b.single_size(), shape_b.size());
    assert!(a.batch_size().is_none());

    let bs = b.batch_size();
    out.set_batch_size(bs)?;

    D::sparse_affine_activate(
        bs.unwrap_or(1),
        stride,
        activation,
        &a.buf,
        shape_a,
        &b.buf,
        shape_b,
        b.nnz,
        c.map(|x| &x.0.buf),
        &mut out.buf,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn backprop_affine_activate<D: Device>(
    stride: Option<bool>,
    activation: Activation,
    a: &mut Tensor<D>,
    shape_a: Shape,
    b: &SparseMatrix<D>,
    shape_b: Shape,
    c: &mut Option<(&mut Tensor<D>, &D::BufferF32)>,
    output: &DenseMatrix<D>,
    output_grad: &DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    let output_shape = shape_a * shape_b;
    if let Some((c, _)) = &c {
        assert_eq!(output_shape.size(), c.values.single_size());
        assert!(c.values.batch_size().is_none());
    }

    assert_eq!(shape_b.cols(), 1);
    assert_eq!(a.values.single_size(), shape_a.size());
    assert_eq!(b.single_size(), shape_b.size());
    assert!(a.values.batch_size().is_none());
    assert_eq!(b.batch_size(), output.batch_size());
    assert_eq!(b.batch_size(), output_grad.batch_size());

    if let Some((c, _)) = &c {
        assert!(c.values.batch_size().is_none());
        assert_eq!(c.values.single_size(), output_shape.size());
    }

    if let Some(agrd) = a.gradients.as_mut() {
        let (c, cgrd) = if let Some((c, _)) = c {
            (Some(&c.values.dense()?.buf), c.gradients.as_mut().map(|g| &mut g.buf))
        } else {
            (None, None)
        };

        D::backprop_sparse_affine_activate(
            output_grad.batch_size().unwrap_or(1),
            stride,
            activation,
            &a.values.dense()?.buf,
            &mut agrd.buf,
            shape_a,
            &b.buf,
            shape_b,
            b.nnz,
            c,
            cgrd,
            &output.buf,
            &output_grad.buf,
        )?;
    } else if let Some((c, ones)) = c {
        if let Some(grad) = c.gradients.as_mut() {
            if stride.is_none() {
                backprop_add_single_scaled(*ones, 1.0, c.values.dense()?, grad, output_grad)?;
            } else {
                return Err(OperationError::UnsupportedOperation("strided backprop add in sparse affine".to_string()));
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn affine_dual<D: Device>(
    w: &DenseMatrix<D>,
    w_shape: Shape,
    s: &SparseMatrix<D>,
    n: &SparseMatrix<D>,
    s_shape: Shape,
    b: &DenseMatrix<D>,
    b_shape: Shape,
    output: &mut DenseMatrix<D>,
    activation: Activation,
) -> Result<(), OperationError<D::DeviceError>> {
    assert!(w.batch_size().is_none());
    assert!(b.batch_size().is_none());
    assert_eq!(s.batch_size(), n.batch_size());
    assert_eq!(s.nnz, n.nnz);
    assert_eq!(w_shape.size(), w.single_size());
    assert_eq!(s_shape.size(), s.single_size());
    assert_eq!(s_shape.size(), n.single_size());
    assert_eq!(b_shape.size(), b.single_size());

    output.set_batch_size(s.batch_size())?;

    affine_activate(Some(false), activation, w, w_shape, s, s_shape, Some((b, b_shape)), output)?;
    affine_activate(Some(true), activation, w, w_shape, n, s_shape, Some((b, b_shape)), output)
}

#[allow(clippy::too_many_arguments)]
pub fn backprop_affine_dual<D: Device>(
    w: &mut Tensor<D>,
    w_shape: Shape,
    s: &SparseMatrix<D>,
    n: &SparseMatrix<D>,
    s_shape: Shape,
    b: &mut Option<(&mut Tensor<D>, &D::BufferF32)>,
    output: &DenseMatrix<D>,
    output_grad: &DenseMatrix<D>,
    activation: Activation,
) -> Result<(), OperationError<D::DeviceError>> {
    assert_eq!(s.batch_size(), n.batch_size());
    assert_eq!(s.nnz, n.nnz);
    assert_eq!(s_shape.size(), s.single_size());
    assert_eq!(s_shape.size(), n.single_size());
    assert_eq!(output.batch_size(), s.batch_size());

    backprop_affine_activate(Some(false), activation, w, w_shape, s, s_shape, b, output, output_grad)?;
    backprop_affine_activate(Some(true), activation, w, w_shape, n, s_shape, b, output, output_grad)?;

    Ok(())
}
