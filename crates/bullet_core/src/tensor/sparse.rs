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
        assert_eq!(shape.cols(), 1);
        Self { buf: D::Buffer::new(device, shape.size()), shape, nnz }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn allocated_size(&self) -> usize {
        self.buf.size()
    }

    pub fn reshape_if_needed(&mut self, shape: Shape, nnz: usize) {
        assert_eq!(shape.cols(), 1, "{shape}");
        let new_size = nnz * shape.batch_size().unwrap_or(1);
        if new_size > self.allocated_size() {
            self.buf = D::Buffer::new(self.buf.device(), new_size);
        } else if self.shape != shape {
            self.buf.set_zero();
        }

        self.shape = shape;
        self.nnz = nnz;
    }

    /// #### Safety
    /// It is the responsibility of the user to ensure all indices fall within the given shape.
    pub unsafe fn load_from_slice(&mut self, shape: Shape, nnz: usize, buf: &[i32]) {
        assert_eq!(nnz * shape.batch_size().unwrap_or(1), buf.len());
        self.reshape_if_needed(shape, nnz);
        self.buf.load_from_slice(buf);
    }

    pub fn copy_into(&self, dest: &mut Self) {
        dest.reshape_if_needed(self.shape, self.nnz);
        dest.buf.load_from_device(&self.buf, self.nnz * self.shape.batch_size().unwrap_or(1));
    }
}
