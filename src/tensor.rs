mod backend;
mod dense_matrix;
mod matrix;
mod shape;
mod sparse_matrix;

pub use backend::{conv::ConvolutionDescription, util, ExecutionContext};
pub use dense_matrix::{Activation, DenseMatrix};
pub use matrix::Matrix;
pub use shape::Shape;
pub use sparse_matrix::SparseMatrix;

use crate::rng;

#[derive(Debug, Default)]
pub struct Tensor {
    pub(crate) values: Matrix,
    pub(crate) gradients: Option<DenseMatrix>,
    pub(crate) internal: Vec<(String, DenseMatrix)>,
}

impl Tensor {
    pub fn new(shape: Shape, requires_grad: bool) -> Self {
        Self {
            values: Matrix::Dense(DenseMatrix::zeroed(shape)),
            gradients: if requires_grad { Some(DenseMatrix::zeroed(shape)) } else { None },
            internal: Vec::new(),
        }
    }

    pub fn zero_grad(&mut self) {
        if let Some(grad) = self.gradients.as_mut() {
            grad.set_zero();
        }
    }

    pub fn copy_values_into(&self, dest: &mut Self) {
        self.values.copy_into(&mut dest.values);
    }

    pub fn get_scalar(&self) -> Option<f32> {
        if self.values.shape() == Shape::new(1, 1) {
            let mut buf = [0.0];
            self.values.dense().write_to_slice(&mut buf);
            Some(buf[0])
        } else {
            None
        }
    }

    pub fn shape(&self) -> Shape {
        self.values.shape()
    }

    pub fn get_dense_vals(&self) -> Option<Vec<f32>> {
        match &self.values {
            Matrix::Sparse(_) => None,
            Matrix::Dense(dense) => {
                let mut buf = vec![0.0; dense.shape.size()];
                dense.write_to_slice(&mut buf);
                Some(buf)
            }
        }
    }

    pub fn set_grad_to_unit(&mut self) {
        let grad = self.gradients.as_mut().unwrap();
        grad.load_from_slice(Shape::new(1, 1), &[1.0]);
    }

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

    pub fn load_dense_from_slice(&mut self, shape: Shape, values: &[f32]) {
        if let Matrix::Dense(dst) = &mut self.values {
            dst.load_from_slice(shape, values);
        } else {
            let mut dst = DenseMatrix::default();
            dst.load_from_slice(shape, values);
            self.values = Matrix::Dense(dst);
        }
    }

    /// #### Safety
    /// It is the responsibility of the user to ensure that all indices fall within the given shape.
    pub unsafe fn load_sparse_from_slice(&mut self, shape: Shape, max_active: usize, values: &[i32]) {
        if let Matrix::Sparse(dst) = &mut self.values {
            dst.load_from_slice(shape, max_active, values);
        } else {
            let mut dst = SparseMatrix::default();
            dst.load_from_slice(shape, max_active, values);
            self.values = Matrix::Sparse(dst);
        }
    }
}
