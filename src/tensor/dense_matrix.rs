mod activate;
mod add;
mod matmul;
mod power_error;

use super::shape::Shape;
use crate::backend::Buffer;

#[derive(Debug)]
pub struct DenseMatrix {
    shape: Shape,
    buf: Buffer<f32>,
}

impl Default for DenseMatrix {
    fn default() -> Self {
        Self::zeroed(Shape::new(1, 1))
    }
}

impl DenseMatrix {
    pub fn zeroed(shape: Shape) -> Self {
        Self { shape, buf: Buffer::new(shape.size()) }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn allocated_size(&self) -> usize {
        self.buf.size()
    }

    /// - If the provided `shape` matches the matrix's current shape, nothing is done.
    /// - If it doesn't, the matrix is reshaped into this shape, and its values are zeroed.
    /// #### WARNING
    /// This is a function for internal use only, with potentially
    /// unintentional side effects.
    fn reshape_if_needed(&mut self, shape: Shape) {
        if shape.size() > self.allocated_size() {
            self.buf = Buffer::new(shape.size());
        } else if self.shape != shape {
            self.buf.set_zero();
        }

        self.shape = shape;
    }

    pub fn load_from_slice(&mut self, shape: Shape, buf: &[f32]) {
        self.reshape_if_needed(shape);
        self.buf.load_from_slice(buf);
    }

    pub fn set_zero(&mut self) {
        self.buf.set_zero();
    }

    pub fn copy_into(&self, dest: &mut Self) {
        dest.reshape_if_needed(self.shape);
        dest.buf.load_from_device(&self.buf);
    }

    /// Writes the contents of this matrix into a buffer,
    /// returns number of values written.
    pub fn write_to_slice(&self, buf: &mut [f32]) -> usize {
        assert!(self.allocated_size() <= buf.len());
        self.buf.write_into_slice(buf);
        self.allocated_size()
    }
}
