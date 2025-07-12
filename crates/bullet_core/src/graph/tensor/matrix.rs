use std::sync::Arc;

use crate::device::{Device, DeviceBuffer, OperationError};

use super::{dense::DenseMatrix, sparse::SparseMatrix};

pub enum Matrix<D: Device> {
    Dense(DenseMatrix<D>),
    Sparse(SparseMatrix<D>),
}

impl<D: Device> Matrix<D> {
    pub fn single_size(&self) -> usize {
        match self {
            Self::Dense(dense) => dense.single_size(),
            Self::Sparse(sparse) => sparse.single_size(),
        }
    }

    pub fn batch_size(&self) -> Option<usize> {
        match self {
            Self::Dense(dense) => dense.batch_size(),
            Self::Sparse(sparse) => sparse.batch_size(),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Self::Dense(dense) => dense.size(),
            Self::Sparse(sparse) => sparse.size(),
        }
    }

    pub fn device(&self) -> Arc<D> {
        match self {
            Self::Dense(dense) => dense.buf.device(),
            Self::Sparse(sparse) => sparse.buf.device(),
        }
    }

    pub fn dense(&self) -> Result<&DenseMatrix<D>, OperationError<D::DeviceError>> {
        if let Self::Dense(matrix) = self {
            Ok(matrix)
        } else {
            Err(OperationError::InvalidTensorFormat)
        }
    }

    pub fn dense_mut(&mut self) -> Result<&mut DenseMatrix<D>, OperationError<D::DeviceError>> {
        if let Self::Dense(matrix) = self {
            Ok(matrix)
        } else {
            Err(OperationError::InvalidTensorFormat)
        }
    }

    pub fn sparse(&self) -> Result<&SparseMatrix<D>, OperationError<D::DeviceError>> {
        if let Self::Sparse(matrix) = self {
            Ok(matrix)
        } else {
            Err(OperationError::InvalidTensorFormat)
        }
    }

    pub fn sparse_mut(&mut self) -> Result<&mut SparseMatrix<D>, OperationError<D::DeviceError>> {
        if let Self::Sparse(matrix) = self {
            Ok(matrix)
        } else {
            Err(OperationError::InvalidTensorFormat)
        }
    }

    pub fn swap_with(&mut self, other: &mut Self) -> Result<(), OperationError<D::DeviceError>> {
        match other {
            Self::Dense(x) => x.swap_with(self.dense_mut()?)?,
            Self::Sparse(x) => x.swap_with(self.sparse_mut()?)?,
        }

        Ok(())
    }

    pub fn copy_into(&self, other: &mut Self) -> Result<(), OperationError<D::DeviceError>> {
        match other {
            Self::Dense(x) => x.copy_from(self.dense()?)?,
            Self::Sparse(x) => x.copy_from(self.sparse()?)?,
        }

        Ok(())
    }
}
