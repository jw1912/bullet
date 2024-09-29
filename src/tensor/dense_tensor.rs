mod add;
mod matmul;

use super::{buffer::Buffer, shape::Shape};

#[derive(Debug)]
pub struct DenseTensor {
    shape: Shape,
    buf: Buffer<f32>,
}

impl Default for DenseTensor {
    fn default() -> Self {
        Self::zeroed(Shape::new(1, 1))
    }
}

impl DenseTensor {
    pub fn zeroed(shape: Shape) -> Self {
        Self {
            shape,
            buf: Buffer::new(shape.size()),
        }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn size(&self) -> usize {
        self.buf.size()
    }

    /// - If the provided `shape` matches the tensor's current shape, nothing is done.
    /// - If it doesn't, the tensor is reshaped into this shape, and its values are zeroed.
    /// #### WARNING
    /// This is a function for internal use only, with potentially
    /// unintentional side effects.
    fn reshape_if_needed(&mut self, shape: Shape) {
        if shape.size() > self.size() {
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

    pub fn write_to_slice(&self, buf: &mut [f32]) -> usize {
        assert!(self.size() <= buf.len());
        self.buf.write_into_slice(buf);
        self.size()
    }
}
