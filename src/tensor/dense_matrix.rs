mod activate;
mod adamw;
mod concat;
mod conv;
mod linear_comb;
mod matmul;
mod pairwise;
mod power_error;
mod slice;
mod softmax;
mod submatrix_product;

use super::{backend::Buffer, shape::Shape};
pub use activate::Activation;

#[derive(Debug)]
pub struct DenseMatrix {
    pub(super) shape: Shape,
    pub(super) buf: Buffer<f32>,
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
    pub(crate) fn reshape_if_needed(&mut self, shape: Shape) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_write_dense_matrix() {
        let mut matrix = DenseMatrix::default();

        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        matrix.load_from_slice(Shape::new(3, 3), &values);

        let bytes = matrix.write_to_byte_buffer("matrix").unwrap();

        println!("{bytes:?}");

        matrix.set_zero();

        let (id, bytes_read) = matrix.read_from_byte_buffer(&bytes);

        assert_eq!(id.as_str(), "matrix");
        assert_eq!(bytes_read, 59);

        let mut buf = [0.0; 9];

        matrix.write_to_slice(&mut buf);

        assert_eq!(buf, values);
    }

    #[test]
    fn attempt_invalid_writes() {
        let matrix = DenseMatrix::default();
        assert!(matrix.write_to_byte_buffer("matrix\n").is_err());
        assert!(matrix.write_to_byte_buffer("màtrix\n").is_err());
    }
}
