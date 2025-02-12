mod dense;
mod matrix;
mod rng;
mod sparse;

use std::{cell::RefCell, collections::HashMap, sync::Arc};

pub use dense::DenseMatrix;
pub use matrix::Matrix;
pub use sparse::SparseMatrix;

use crate::{
    device::Device,
    graph::{builder::Node, operation::Operation},
};

pub struct Tensor<D: Device> {
    pub values: Matrix<D>,
    pub gradients: Option<DenseMatrix<D>>,
    pub(crate) internal: HashMap<String, RefCell<DenseMatrix<D>>>,
    pub(crate) operation: Option<Operation>,
    pub(crate) own: Node,
}

impl<D: Device> Tensor<D> {
    pub fn new(
        device: Arc<D>,
        single_size: usize,
        requires_grad: bool,
        operation: Option<Operation>,
        own: Node,
    ) -> Self {
        Self {
            values: Matrix::Dense(DenseMatrix::zeroed(device.clone(), single_size)),
            gradients: requires_grad.then(|| DenseMatrix::zeroed(device, single_size)),
            internal: HashMap::new(),
            operation,
            own,
        }
    }

    pub fn zero_grad(&mut self) {
        if let Some(grad) = self.gradients.as_mut() {
            grad.set_zero();
        }
    }

    pub fn get_scalar(&self) -> Option<f32> {
        if self.values.size() == 1 {
            let mut buf = [0.0];
            self.values.dense().write_to_slice(&mut buf);
            Some(buf[0])
        } else {
            None
        }
    }

    pub fn get_dense_vals(&self) -> Option<Vec<f32>> {
        match &self.values {
            Matrix::Sparse(_) => None,
            Matrix::Dense(dense) => {
                let mut buf = vec![0.0; dense.size()];
                dense.write_to_slice(&mut buf);
                Some(buf)
            }
        }
    }

    pub fn set_grad_to_unit(&mut self) {
        let grad = self.gradients.as_mut().unwrap();
        grad.load_from_slice(None, &[1.0]);
    }

    pub fn load_from_slice(&mut self, batch_size: Option<usize>, values: &[f32]) {
        if let Matrix::Dense(dst) = &mut self.values {
            assert_eq!(values.len(), dst.size());
            dst.load_from_slice(batch_size, values);
        } else {
            panic!("This tensor is sparse!")
        }
    }

    pub fn seed_random(&mut self, mean: f32, stdev: f32, use_gaussian: bool) {
        let values = rng::vec_f32(self.values.size(), mean, stdev, use_gaussian);
        self.load_from_slice(self.values.batch_size(), &values);
    }

    pub fn load_dense_from_slice(&mut self, batch_size: Option<usize>, values: &[f32]) {
        if let Matrix::Dense(dst) = &mut self.values {
            dst.load_from_slice(batch_size, values);
        } else {
            let mut dst = DenseMatrix::zeroed(self.values.device(), self.values.single_size());
            dst.load_from_slice(batch_size, values);
            self.values = Matrix::Dense(dst);
        }
    }

    /// #### Safety
    /// It is the responsibility of the user to ensure that all indices fall within the given shape.
    pub unsafe fn load_sparse_from_slice(&mut self, nnz: usize, batch_size: Option<usize>, values: &[i32]) {
        if let Matrix::Sparse(dst) = &mut self.values {
            dst.load_from_slice(nnz, batch_size, values);
        } else {
            let mut dst = SparseMatrix::zeroed(self.values.device(), self.values.single_size(), nnz);
            dst.load_from_slice(nnz, batch_size, values);
            self.values = Matrix::Sparse(dst);
        }
    }
}
