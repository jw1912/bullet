use std::sync::Arc;

use crate::{
    device::{Device, DeviceBuffer},
    tensor::{dense::DenseMatrix, sparse::SparseMatrix},
};

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

    pub fn dense(&self) -> &DenseMatrix<D> {
        if let Self::Dense(matrix) = self {
            matrix
        } else {
            panic!("This matrix is not dense!")
        }
    }

    pub fn dense_mut(&mut self) -> &mut DenseMatrix<D> {
        if let Self::Dense(matrix) = self {
            matrix
        } else {
            panic!("This matrix is not dense!")
        }
    }

    pub fn sparse(&self) -> &SparseMatrix<D> {
        if let Self::Sparse(matrix) = self {
            matrix
        } else {
            panic!("This matrix is not sparse!")
        }
    }
}
