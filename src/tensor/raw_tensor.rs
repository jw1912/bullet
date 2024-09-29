mod add;
mod linear;
mod mul;

use super::{buffer::Buffer, shape::Shape};

#[derive(Debug)]
pub enum RawTensorValues {
    Empty,
    DenseFloats(Buffer<f32>),
    SparseUnits {
        max_active: usize,
        buf: Buffer<i32>,
    },
}

#[derive(Debug)]
pub struct RawTensor {
    shape: Shape,
    len: usize,
    values: RawTensorValues,
}

impl Default for RawTensor {
    fn default() -> Self {
        Self {
            shape: Shape::new(0, 0),
            len: 0,
            values: RawTensorValues::Empty,
        }
    }
}

impl RawTensor {
    pub fn new_empty(shape: Shape) -> Self {
        Self {
            shape,
            len: 0,
            values: RawTensorValues::Empty,
        }
    }

    pub fn new_dense(shape: Shape, cap: usize) -> Self {
        Self {
            shape,
            len: 0,
            values: RawTensorValues::DenseFloats(Buffer::new(cap * shape.size())),
        }
    }

    pub fn new_sparse(shape: Shape, cap: usize, max_active: usize) -> Self {
        Self {
            shape,
            len: 0,
            values: RawTensorValues::SparseUnits {
                max_active,
                buf: Buffer::new(cap * max_active)
            },
        }
    }

    pub fn cap(&self) -> usize {
        match &self.values {
            RawTensorValues::Empty => 0,
            RawTensorValues::SparseUnits { max_active, buf } => buf.size() / max_active,
            RawTensorValues::DenseFloats(buf) => buf.size() / self.shape.size(),
        }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn size(&self) -> usize {
        self.len * self.shape.size()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape == Shape::new(1, 1) && self.len == 1
    }

    pub fn set_zero(&mut self) {
        match &mut self.values {
            RawTensorValues::Empty => {
                panic!("Cannot zero an empty tensor, we do not know its layout!");
            }
            RawTensorValues::DenseFloats(buf) => {
                buf.set_zero();
            }
            RawTensorValues::SparseUnits { buf, .. } => {
                buf.set_zero();
            }
        }
    }

    pub fn resize_if_necessary(&mut self, len: usize) {
        if len > self.cap() {
            match &mut self.values {
                RawTensorValues::Empty => {
                    panic!("Cannot resize an empty tensor, we do not know its layout!");
                }
                RawTensorValues::DenseFloats(buf) => {
                    *buf = Buffer::new(len * self.shape.size());
                }
                RawTensorValues::SparseUnits{ buf, max_active } => {
                    *buf = Buffer::new(len * *max_active);
                }
            }
        }
    }

    pub fn copy_values_into(&self, dest: &mut Self) {
        if let RawTensorValues::Empty = dest.values {
            match &self.values {
                RawTensorValues::Empty => {
                    panic!("Attempting to copy from an empty tensor!")
                }
                RawTensorValues::DenseFloats(_) => {
                    dest.values = RawTensorValues::DenseFloats(Buffer::new(self.size()))
                }
                RawTensorValues::SparseUnits{ max_active, .. } => {
                    dest.values = RawTensorValues::SparseUnits {
                        max_active: *max_active,
                        buf: Buffer::new(self.size()),
                    }
                }
            }
        }

        assert_eq!(self.shape, dest.shape);
        dest.resize_if_necessary(self.len);
        dest.len = self.len;
        
        match (&self.values, &dest.values) {
            (RawTensorValues::DenseFloats(src), RawTensorValues::DenseFloats(dst)) => {
                dst.load_from_device(src);
            }
            (
                RawTensorValues::SparseUnits { max_active: src_act, buf: src },
                RawTensorValues::SparseUnits { max_active: dst_act, buf: dst },
            ) => {
                assert_eq!(src_act, dst_act);
                dst.load_from_device(src);
            }
            _ => panic!("Invalid combination of tensors to copy across!")
        }
    }

    pub fn load_sparse_from_slice(&mut self, max_active: usize, buf: &[i32]) {
        if let RawTensorValues::Empty = self.values {
            self.values = RawTensorValues::SparseUnits {
                max_active,
                buf: Buffer::new(buf.len()),
            };
        }

        assert_eq!(
            buf.len() % self.shape.size(),
            0,
            "Buffer does not contain an integer number of tensors!",
        );

        let len = buf.len() / self.shape.size();
        self.resize_if_necessary(len);
        self.len = len;

        if let RawTensorValues::SparseUnits { buf: dest, max_active: max } = &self.values {
            assert_eq!(*max, max_active);
            dest.load_from_slice(buf);
        } else {
            panic!("This tensor does not have sparse layout!");
        }
    }

    pub fn write_sparse_to_slice(&self, buf: &mut [i32]) -> usize {
        if let RawTensorValues::SparseUnits { buf: dest, max_active } = &self.values {
            dest.write_into_slice(buf);
            *max_active
        } else {
            panic!("This tensor does not have sparse layout!");
        }
    }

    pub fn load_dense_from_slice(&mut self, buf: &[f32]) {
        if let RawTensorValues::Empty = self.values {
            self.values = RawTensorValues::DenseFloats(Buffer::new(buf.len()));
        }

        assert_eq!(
            buf.len() % self.shape.size(),
            0,
            "Buffer does not contain an integer number of tensors!",
        );

        let len = buf.len() / self.shape.size();
        self.resize_if_necessary(len);
        self.len = len;

        if let RawTensorValues::DenseFloats(dest) = &self.values {
            dest.load_from_slice(buf);
        } else {
            panic!("This tensor does not have dense layout!");
        }
    }

    pub fn write_dense_to_slice(&self, buf: &mut [f32]) {
        if let RawTensorValues::DenseFloats(dest) = &self.values {
            dest.write_into_slice(buf);
        } else {
            panic!("This tensor does not have dense layout!");
        }
    }
}