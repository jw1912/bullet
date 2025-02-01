use std::sync::Arc;

use crate::{
    device::{Device, DeviceBuffer},
    shape::Shape,
};

pub struct SparseMatrix<D: Device> {
    pub buf: D::Buffer<i32>,
    pub shape: Shape,
    pub nnz: usize,
}

impl<D: Device> SparseMatrix<D> {
    pub fn zeroed(device: Arc<D>, shape: Shape, nnz: usize) -> Self {
        Self { buf: D::Buffer::new(device, shape.size()), shape, nnz }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn allocated_size(&self) -> usize {
        self.buf.size()
    }

    pub fn reshape_if_needed(&mut self, shape: Shape, nnz: usize) {
        if nnz * shape.cols() > self.allocated_size() {
            self.buf = D::Buffer::new(self.buf.device(), shape.size());
        } else if self.shape != shape {
            self.buf.set_zero();
        }

        self.shape = shape;
        self.nnz = nnz;
    }

    /// #### Safety
    /// It is the responsibility of the user to ensure all indices fall within the given shape.
    pub unsafe fn load_from_slice(&mut self, shape: Shape, max_active: usize, buf: &[i32]) {
        self.reshape_if_needed(shape, max_active);
        self.buf.load_from_slice(buf);
    }

    pub fn copy_into(&self, dest: &mut Self) {
        dest.reshape_if_needed(self.shape, self.nnz);
        dest.buf.load_from_device(&self.buf, self.nnz * self.shape.cols());
    }
}
