use std::sync::Arc;

use crate::{
    device::{Device, DeviceBuffer},
    shape::Shape,
};

pub struct DenseMatrix<D: Device> {
    pub buf: D::Buffer<f32>,
    pub shape: Shape,
}

impl<D: Device> DenseMatrix<D> {
    pub fn zeroed(device: Arc<D>, shape: Shape) -> Self {
        Self { buf: D::Buffer::new(device, shape.size()), shape }
    }

    pub fn ones(device: Arc<D>, shape: Shape) -> Self {
        let mut res = Self { buf: D::Buffer::new(device, shape.size()), shape };
        res.load_from_slice(shape, &vec![1.0; shape.size()]);
        res
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
    pub fn reshape_if_needed(&mut self, shape: Shape) {
        if shape.size() > self.allocated_size() {
            self.buf = D::Buffer::new(self.buf.device(), shape.size());
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
        dest.buf.load_from_device(&self.buf, self.shape.size());
    }

    /// Writes the contents of this matrix into a buffer,
    /// returns number of values written.
    pub fn write_to_slice(&self, buf: &mut [f32]) -> usize {
        assert!(self.shape.size() <= buf.len());
        self.buf.write_into_slice(buf, self.shape.size());
        self.allocated_size()
    }

    /// Writes a complete description of the matrix into a buffer of bytes,
    /// along with an ID tag.
    pub fn write_to_byte_buffer(&self, id: &str) -> std::io::Result<Vec<u8>> {
        use std::io::{Error, ErrorKind, Write};

        if !id.is_ascii() {
            return Err(Error::new(ErrorKind::InvalidInput, "IDs may not contain non-ASCII characters!"));
        }

        if id.contains('\n') {
            return Err(Error::new(ErrorKind::InvalidInput, "IDs may not contain newlines!"));
        }

        let mut id_bytes = id.chars().map(|ch| ch as u8).collect::<Vec<_>>();

        id_bytes.push(b'\n');

        let mut buf = Vec::new();

        buf.write_all(&id_bytes)?;
        buf.write_all(&usize::to_le_bytes(self.shape.rows()))?;
        buf.write_all(&usize::to_le_bytes(self.shape.cols()))?;

        let mut values = vec![0.0; self.shape.size()];
        self.write_to_slice(&mut values);

        for val in values {
            buf.write_all(&f32::to_le_bytes(val))?;
        }

        Ok(buf)
    }

    /// Reads a matrix from a byte buffer, returning how many bytes were read
    /// and the matrix ID that was read.
    pub fn read_from_byte_buffer(&mut self, bytes: &[u8]) -> (String, usize) {
        const USIZE: usize = std::mem::size_of::<usize>();

        let mut offset = 0;

        let mut id = String::new();
        loop {
            let ch = bytes[offset];
            offset += 1;

            if ch == b'\n' {
                break;
            }

            id.push(char::from(ch));
        }

        let mut rows = [0u8; USIZE];
        rows.copy_from_slice(&bytes[offset..offset + USIZE]);
        offset += USIZE;

        let mut cols = [0u8; USIZE];
        cols.copy_from_slice(&bytes[offset..offset + USIZE]);
        offset += USIZE;

        let rows = usize::from_le_bytes(rows);
        let cols = usize::from_le_bytes(cols);

        let shape = Shape::new(rows, cols);
        let total_read = offset + shape.size() * 4;

        let mut values = vec![0.0; shape.size()];

        for (word, val) in bytes[offset..total_read].chunks_exact(4).zip(values.iter_mut()) {
            let mut buf = [0; 4];
            buf.copy_from_slice(word);
            *val = f32::from_le_bytes(buf);
        }

        self.load_from_slice(shape, &values);

        (id, total_read)
    }
}
