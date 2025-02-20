use crate::{
    device::{Device, OperationError},
    shape::Shape,
    tensor::{DenseMatrix, SparseMatrix},
};

use super::Activation;

#[allow(clippy::too_many_arguments)]
pub fn sparse_affine_dual<D: Device>(
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

    D::sparse_affine_dual_activate(
        s.batch_size().unwrap_or(1),
        &w.buf,
        w_shape,
        &s.buf,
        &n.buf,
        s_shape,
        s.nnz,
        &b.buf,
        &mut output.buf,
        activation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn backprop_sparse_affine_dual_activate<D: Device>(
    w: &DenseMatrix<D>,
    wgrd: Option<&mut DenseMatrix<D>>,
    w_shape: Shape,
    s: &SparseMatrix<D>,
    n: &SparseMatrix<D>,
    s_shape: Shape,
    b: &DenseMatrix<D>,
    bgrd: Option<&mut DenseMatrix<D>>,
    b_shape: Shape,
    output: &DenseMatrix<D>,
    output_grad: &DenseMatrix<D>,
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
    assert_eq!(output.batch_size(), s.batch_size());

    if let Some(wgrd) = wgrd {
        assert_eq!(wgrd.single_size(), w.single_size());

        D::backprop_sparse_affine_dual_activate(
            output.batch_size().unwrap_or(1),
            &w.buf,
            &mut wgrd.buf,
            w_shape,
            &s.buf,
            &n.buf,
            s_shape,
            s.nnz,
            &b.buf,
            &mut bgrd.unwrap().buf,
            &output.buf,
            &output_grad.buf,
            activation,
        )?;
    }

    Ok(())
}
