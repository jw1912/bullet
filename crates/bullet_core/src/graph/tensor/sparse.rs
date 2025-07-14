use std::{num::NonZeroUsize, sync::Arc};

use crate::device::{Device, DeviceBuffer, OperationError};

use super::DenseMatrix;

pub struct SparseMatrix<D: Device> {
    pub buf: D::BufferI32,
    pub nnz: usize,
    pub single_size: usize,
    pub batch_size: Option<NonZeroUsize>,
}

impl<D: Device> SparseMatrix<D> {
    pub fn zeroed(device: Arc<D>, single_size: usize, nnz: usize) -> Result<Self, D::DeviceError> {
        Ok(Self { buf: D::BufferI32::new(device, nnz)?, single_size, nnz, batch_size: None })
    }

    pub fn allocated_size(&self) -> usize {
        self.buf.size()
    }

    pub fn set_batch_size(&mut self, batch_size: Option<usize>) -> Result<(), D::DeviceError> {
        let new_size = self.nnz * batch_size.unwrap_or(1);
        if new_size > self.allocated_size() {
            self.buf = D::BufferI32::new(self.buf.device(), new_size)?;
        } else if batch_size != self.batch_size() {
            self.buf.set_zero()?;
        }

        self.batch_size = batch_size.map(|x| NonZeroUsize::new(x).unwrap());

        Ok(())
    }

    pub fn swap_with(&mut self, other: &mut Self) -> Result<(), D::DeviceError> {
        if self.single_size != other.single_size || self.nnz != other.nnz {
            return Err(D::DeviceError::default());
        }

        std::mem::swap(self, other);

        Ok(())
    }

    pub fn copy_from(&mut self, other: &Self) -> Result<(), D::DeviceError> {
        if self.single_size != other.single_size || self.nnz != other.nnz {
            return Err(D::DeviceError::default());
        }

        self.set_batch_size(other.batch_size())?;
        self.buf.load_from_device(&other.buf, other.nnz * other.batch_size().unwrap_or(1))
    }

    pub fn single_size(&self) -> usize {
        self.single_size
    }

    pub fn batch_size(&self) -> Option<usize> {
        self.batch_size.map(NonZeroUsize::get)
    }

    pub fn size(&self) -> usize {
        self.single_size * self.batch_size().unwrap_or(1)
    }

    /// #### Safety
    /// It is the responsibility of the user to ensure all indices fall within the given shape.
    pub unsafe fn load_from_slice(
        &mut self,
        nnz: usize,
        batch_size: Option<usize>,
        buf: &[i32],
    ) -> Result<(), D::DeviceError> {
        if self.nnz != nnz || nnz * batch_size.unwrap_or(1) != buf.len() {
            return Err(D::DeviceError::default());
        }

        self.set_batch_size(batch_size)?;
        self.buf.load_from_slice(buf)
    }

    /// #### Safety
    /// It is the responsibility of the user to ensure all indices fall within the given shape.
    ///
    /// Must synchronise before `buf` is dropped or mutated.
    pub unsafe fn load_non_blocking_from_host(
        &mut self,
        nnz: usize,
        batch_size: Option<usize>,
        buf: &[i32],
    ) -> Result<(), D::DeviceError> {
        if self.nnz != nnz || nnz * batch_size.unwrap_or(1) != buf.len() {
            return Err(D::DeviceError::default());
        }

        self.set_batch_size(batch_size)?;
        self.buf.load_non_blocking_from_host(buf)
    }

    pub fn copy_into_dense(&self, dst: &mut DenseMatrix<D>) -> Result<(), OperationError<D::DeviceError>> {
        let batch_size = self.batch_size();
        let size = self.single_size();

        if size != dst.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        dst.set_batch_size(batch_size)?;
        D::sparse_to_dense(batch_size.unwrap_or(1), size, self.nnz, &self.buf, &mut dst.buf)
    }
}
