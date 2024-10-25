mod affine;
mod affine_dual;
mod select;

use super::shape::Shape;
use crate::backend::Buffer;

#[derive(Debug)]
pub struct SparseMatrix {
    shape: Shape,
    max_active: usize,
    buf: Buffer<i32>,
}

impl Default for SparseMatrix {
    fn default() -> Self {
        Self::zeroed(Shape::new(1, 1), 1)
    }
}

impl SparseMatrix {
    pub fn zeroed(shape: Shape, max_active: usize) -> Self {
        Self { shape, max_active, buf: Buffer::new(max_active * shape.cols()) }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn allocated_size(&self) -> usize {
        self.buf.size()
    }

    fn reshape_if_needed(&mut self, shape: Shape, max_active: usize) {
        if max_active * shape.cols() > self.allocated_size() {
            self.buf = Buffer::new(shape.size());
        } else if self.shape != shape {
            self.buf.set_zero();
        }

        self.shape = shape;
        self.max_active = max_active;
    }

    /// #### Safety
    /// It is the responsibility of the user to ensure all indices fall within the given shape.
    pub unsafe fn load_from_slice(&mut self, shape: Shape, max_active: usize, buf: &[i32]) {
        self.reshape_if_needed(shape, max_active);
        self.buf.load_from_slice(buf);
    }

    pub fn copy_into(&self, dest: &mut Self) {
        dest.reshape_if_needed(self.shape, self.max_active);
        dest.buf.load_from_device(&self.buf, self.max_active * self.shape.cols());
    }
}
