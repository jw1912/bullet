mod dense_matrix;
mod matrix;
mod shape;
mod sparse_matrix;

pub use dense_matrix::DenseMatrix;
pub use matrix::Matrix;
pub use shape::Shape;
pub use sparse_matrix::SparseMatrix;

use crate::{backend::ExecutionContext, operations::Operation};

impl From<Tensor> for Shape {
    fn from(value: Tensor) -> Self {
        value.values.shape()
    }
}

#[derive(Debug, Default)]
pub struct Tensor {
    pub(crate) values: Matrix,
    pub(crate) gradients: Option<DenseMatrix>,
}

impl diffable::Tensor for Tensor {
    type ModelOfTensor = Shape;
    type ExecutionContext = ExecutionContext;
    type DiffableOperation = Operation;

    fn new(shape: Shape, requires_grad: bool) -> Self {
        Self {
            values: Matrix::Dense(DenseMatrix::zeroed(shape)),
            gradients: if requires_grad { Some(DenseMatrix::zeroed(shape)) } else { None },
        }
    }

    fn zero_grad(&mut self) {
        if let Some(grad) = self.gradients.as_mut() {
            grad.set_zero();
        }
    }

    fn copy_values_into(&self, dest: &mut Self) {
        self.values.copy_into(&mut dest.values);
    }

    fn get_scalar(&self) -> Option<f32> {
        if self.values.shape() == Shape::new(1, 1) {
            let mut buf = [0.0];
            self.values.dense().write_to_slice(&mut buf);
            Some(buf[0])
        } else {
            None
        }
    }

    fn set_grad_to_unit(&mut self) {
        let grad = self.gradients.as_mut().unwrap();
        grad.load_from_slice(Shape::new(1, 1), &[1.0]);
    }
}

impl Tensor {
    pub fn load_dense_from_slice(&mut self, shape: Shape, values: &[f32]) {
        if let Matrix::Dense(dst) = &mut self.values {
            dst.load_from_slice(shape, values);
        } else {
            let mut dst = DenseMatrix::default();
            dst.load_from_slice(shape, values);
            self.values = Matrix::Dense(dst);
        }
    }

    pub fn load_sparse_from_slice(&mut self, shape: Shape, max_active: usize, values: &[i32]) {
        if let Matrix::Sparse(dst) = &mut self.values {
            dst.load_from_slice(shape, max_active, values);
        } else {
            let mut dst = SparseMatrix::default();
            dst.load_from_slice(shape, max_active, values);
            self.values = Matrix::Sparse(dst);
        }
    }
}
