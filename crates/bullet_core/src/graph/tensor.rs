mod dense;
mod matrix;
pub mod rng;
mod sparse;

use std::{num::NonZeroUsize, sync::Arc};

pub use dense::{read_from_byte_buffer, DenseMatrix};
pub use matrix::Matrix;
pub use sparse::SparseMatrix;

use crate::{
    device::{Device, DeviceBuffer, OperationError},
    graph::ir::shape::Shape,
};

pub struct Tensor<D: Device> {
    pub values: Matrix<D>,
    pub shape: Shape,
}

impl<D: Device> Tensor<D> {
    pub fn new(device: Arc<D>, shape: Shape, sparse: Option<NonZeroUsize>) -> Result<Self, D::DeviceError> {
        let single_size = shape.size();

        let values = if let Some(nnz) = sparse.map(usize::from) {
            Matrix::Sparse(SparseMatrix::zeroed(device.clone(), single_size, nnz)?)
        } else {
            Matrix::Dense(DenseMatrix::zeroed(device.clone(), single_size)?)
        };

        Ok(Self { values, shape })
    }

    pub fn zero(&mut self) -> Result<(), OperationError<D::DeviceError>> {
        self.values.dense_mut()?.set_to(0.0)?;
        Ok(())
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn get_scalar(&self) -> Option<f32> {
        if self.values.size() == 1 {
            let mut buf = [0.0];
            self.values.dense().ok()?.write_to_slice(&mut buf).unwrap();
            Some(buf[0])
        } else {
            None
        }
    }

    pub fn get_dense_vals(&self) -> Result<Vec<f32>, OperationError<D::DeviceError>> {
        match &self.values {
            Matrix::Sparse(_) => Err(OperationError::InvalidTensorFormat),
            Matrix::Dense(dense) => {
                let mut buf = vec![0.0; dense.size()];
                dense.write_to_slice(&mut buf)?;
                Ok(buf)
            }
        }
    }

    pub fn get_sparse_vals(&self) -> Result<Vec<i32>, OperationError<D::DeviceError>> {
        match &self.values {
            Matrix::Sparse(sparse) => {
                let size = sparse.nnz * sparse.batch_size().unwrap_or(1);
                let mut buf = vec![0; size];
                sparse.buf.write_into_slice(&mut buf, size)?;
                Ok(buf)
            }
            Matrix::Dense(_) => Err(OperationError::InvalidTensorFormat),
        }
    }

    pub fn seed_random(
        &mut self,
        mean: f32,
        stdev: f32,
        use_gaussian: bool,
    ) -> Result<(), OperationError<D::DeviceError>> {
        let values = rng::vec_f32(self.values.size(), mean, stdev, use_gaussian);
        self.load_dense_from_slice(self.values.batch_size(), &values)
    }

    pub fn dense(&self) -> Result<&DenseMatrix<D>, OperationError<D::DeviceError>> {
        match &self.values {
            Matrix::Dense(dense) => Ok(dense),
            Matrix::Sparse(_) => Err(OperationError::InvalidTensorFormat),
        }
    }

    pub fn dense_mut(&mut self) -> Result<&mut DenseMatrix<D>, OperationError<D::DeviceError>> {
        match &mut self.values {
            Matrix::Dense(dense) => Ok(dense),
            Matrix::Sparse(_) => Err(OperationError::InvalidTensorFormat),
        }
    }

    pub fn sparse(&self) -> Result<&SparseMatrix<D>, OperationError<D::DeviceError>> {
        match &self.values {
            Matrix::Dense(_) => Err(OperationError::InvalidTensorFormat),
            Matrix::Sparse(sparse) => Ok(sparse),
        }
    }

    pub fn sparse_mut(&mut self) -> Result<&mut SparseMatrix<D>, OperationError<D::DeviceError>> {
        match &mut self.values {
            Matrix::Dense(_) => Err(OperationError::InvalidTensorFormat),
            Matrix::Sparse(sparse) => Ok(sparse),
        }
    }

    pub fn load_dense_from_slice(
        &mut self,
        batch_size: Option<usize>,
        values: &[f32],
    ) -> Result<(), OperationError<D::DeviceError>> {
        self.values.dense_mut()?.load_from_slice(batch_size, values)?;
        Ok(())
    }

    /// #### Safety
    /// It is the responsibility of the user to ensure that all indices fall within the given shape.
    pub unsafe fn load_sparse_from_slice(
        &mut self,
        nnz: usize,
        batch_size: Option<usize>,
        values: &[i32],
    ) -> Result<(), OperationError<D::DeviceError>> {
        self.values.sparse_mut()?.load_from_slice(nnz, batch_size, values)?;
        Ok(())
    }
}
