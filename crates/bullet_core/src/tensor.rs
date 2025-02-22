mod dense;
mod matrix;
mod rng;
mod sparse;

use std::{cell::RefCell, collections::HashMap, sync::Arc};

pub use dense::DenseMatrix;
pub use matrix::Matrix;
pub use sparse::SparseMatrix;

use crate::{
    device::{Device, DeviceBuffer, OperationError},
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
    ) -> Result<Self, D::DeviceError> {
        let values = if let Some(nnz) = own.sparse.map(usize::from) {
            Matrix::Sparse(SparseMatrix::zeroed(device.clone(), single_size, nnz)?)
        } else {
            Matrix::Dense(DenseMatrix::zeroed(device.clone(), single_size)?)
        };

        Ok(Self {
            values,
            gradients: if requires_grad { Some(DenseMatrix::zeroed(device, single_size)?) } else { None },
            internal: HashMap::new(),
            operation,
            own,
        })
    }

    pub fn zero_grad(&mut self) -> Result<(), D::DeviceError> {
        if let Some(grad) = self.gradients.as_mut() {
            grad.set_zero()?;
        }

        Ok(())
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

    pub fn set_grad_to_unit(&mut self) -> Result<(), D::DeviceError> {
        let grad = self.gradients.as_mut().unwrap();
        grad.load_from_slice(None, &[1.0])
    }

    pub fn load_from_slice(&mut self, batch_size: Option<usize>, values: &[f32]) -> Result<(), D::DeviceError> {
        if let Matrix::Dense(dst) = &mut self.values {
            assert_eq!(values.len(), dst.size());
            dst.load_from_slice(batch_size, values)
        } else {
            panic!("This tensor is sparse!")
        }
    }

    pub fn seed_random(&mut self, mean: f32, stdev: f32, use_gaussian: bool) -> Result<(), D::DeviceError> {
        let values = rng::vec_f32(self.values.size(), mean, stdev, use_gaussian);
        self.load_from_slice(self.values.batch_size(), &values)
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
