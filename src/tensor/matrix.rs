use super::{DenseMatrix, Shape, SparseMatrix};

#[derive(Debug)]
pub enum Matrix {
    Dense(DenseMatrix),
    Sparse(SparseMatrix),
}

impl Default for Matrix {
    fn default() -> Self {
        Self::Dense(DenseMatrix::default())
    }
}

impl Matrix {
    pub fn shape(&self) -> Shape {
        match self {
            Self::Dense(dense) => dense.shape(),
            Self::Sparse(sparse) => sparse.shape(),
        }
    }

    pub fn dense(&self) -> &DenseMatrix {
        if let Self::Dense(matrix) = self {
            matrix
        } else {
            panic!("This matrix is not dense!")
        }
    }

    pub fn dense_mut(&mut self) -> &mut DenseMatrix {
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
                    let mut dst = DenseMatrix::default();
                    src.copy_into(&mut dst);
                    *dest = Self::Dense(dst);
                }
            },
            Self::Sparse(src) => {
                if let Self::Sparse(dst) = dest {
                    src.copy_into(dst)
                } else {
                    let mut dst = SparseMatrix::default();
                    src.copy_into(&mut dst);
                    *dest = Self::Sparse(dst);
                }
            },
        }
    }
}
