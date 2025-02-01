use std::sync::Arc;

use crate::{
    device::{Device, DeviceBuffer},
    shape::Shape,
    tensor::{dense::DenseMatrix, sparse::SparseMatrix},
};

pub enum Matrix<D: Device> {
    Dense(DenseMatrix<D>),
    Sparse(SparseMatrix<D>),
}

impl<D: Device> Matrix<D> {
    pub fn shape(&self) -> Shape {
        match self {
            Self::Dense(dense) => dense.shape(),
            Self::Sparse(sparse) => sparse.shape(),
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

    pub fn copy_into(&self, dest: &mut Self) {
        match self {
            Self::Dense(src) => {
                if let Self::Dense(dst) = dest {
                    src.copy_into(dst)
                } else {
                    let mut dst = DenseMatrix::zeroed(src.buf.device(), src.shape());
                    src.copy_into(&mut dst);
                    *dest = Self::Dense(dst);
                }
            }
            Self::Sparse(src) => {
                if let Self::Sparse(dst) = dest {
                    src.copy_into(dst)
                } else {
                    let mut dst = SparseMatrix::zeroed(src.buf.device(), src.shape, src.nnz);
                    src.copy_into(&mut dst);
                    *dest = Self::Sparse(dst);
                }
            }
        }
    }
}
