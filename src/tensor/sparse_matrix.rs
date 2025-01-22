mod affine;
mod affine_dual;
mod gather;
mod mask;
mod select;
mod softmax;

use super::{
    backend::{ops, Buffer},
    shape::Shape,
    DenseMatrix,
};

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

    pub fn copy_into_dense(&self, dest: &mut DenseMatrix) {
        dest.reshape_if_needed(self.shape);
        dest.set_zero();

        unsafe {
            ops::sparse_to_dense(
                self.shape.rows(),
                self.shape.cols(),
                self.max_active,
                self.buf.ptr(),
                dest.buf.mut_ptr(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::backend::util;

    use super::*;

    #[test]
    fn sparse_to_dense() {
        let shape = Shape::new(3, 3);

        let mut input = SparseMatrix::default();
        let mut output = DenseMatrix::default();

        util::panic_if_device_error("Failed to initialise matrices!");

        unsafe {
            input.load_from_slice(shape, 2, &[0, -1, 1, 2, 1, -1]);
        }

        input.copy_into_dense(&mut output);

        let mut buf = [0.0; 9];
        output.write_to_slice(&mut buf);

        assert_eq!(buf, [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    }
}
