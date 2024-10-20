mod dense_matrix;
mod matrix;
mod operation;
mod shape;
mod sparse_matrix;

pub use dense_matrix::DenseMatrix;
pub use matrix::Matrix;
pub use operation::Activation;
pub use shape::Shape;
pub use sparse_matrix::SparseMatrix;

pub(crate) use operation::Operation;

use crate::{backend::ExecutionContext, rng};

impl From<Tensor> for Shape {
    fn from(value: Tensor) -> Self {
        value.values.shape()
    }
}

#[derive(Debug, Default)]
pub struct Tensor {
    pub(crate) values: Matrix,
    pub(crate) gradients: Option<DenseMatrix>,
    internal: Vec<(String, DenseMatrix)>,
}

impl diffable::Tensor for Tensor {
    type ModelOfTensor = Shape;
    type ExecutionContext = ExecutionContext;
    type DiffableOperation = Operation;

    fn new(shape: Shape, requires_grad: bool) -> Self {
        Self {
            values: Matrix::Dense(DenseMatrix::zeroed(shape)),
            gradients: if requires_grad { Some(DenseMatrix::zeroed(shape)) } else { None },
            internal: Vec::new(),
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
    pub fn load_from_slice(&mut self, values: &[f32]) {
        if let Matrix::Dense(dst) = &mut self.values {
            assert_eq!(values.len(), dst.shape.size());
            dst.load_from_slice(dst.shape, values);
        } else {
            panic!("This tensor is sparse!")
        }
    }

    pub fn seed_random(&mut self, mean: f32, stdev: f32, use_gaussian: bool) {
        let values = rng::vec_f32(self.values.shape().size(), mean, stdev, use_gaussian);
        self.load_from_slice(&values);
    }

    pub(crate) fn load_dense_from_slice(&mut self, shape: Shape, values: &[f32]) {
        if let Matrix::Dense(dst) = &mut self.values {
            dst.load_from_slice(shape, values);
        } else {
            let mut dst = DenseMatrix::default();
            dst.load_from_slice(shape, values);
            self.values = Matrix::Dense(dst);
        }
    }

    pub(crate) fn load_sparse_from_slice(&mut self, shape: Shape, max_active: usize, values: &[i32]) {
        if let Matrix::Sparse(dst) = &mut self.values {
            dst.load_from_slice(shape, max_active, values);
        } else {
            let mut dst = SparseMatrix::default();
            dst.load_from_slice(shape, max_active, values);
            self.values = Matrix::Sparse(dst);
        }
    }
}
