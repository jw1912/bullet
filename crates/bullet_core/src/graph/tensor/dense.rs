use std::{fmt::Debug, num::NonZeroUsize, sync::Arc};

use crate::device::{base::BaseOperations, Device, DeviceBuffer, OperationError};

pub struct DenseMatrix<D: Device> {
    pub buf: D::BufferF32,
    pub single_size: usize,
    pub batch_size: Option<NonZeroUsize>,
}

impl<D: Device> Debug for DenseMatrix<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{:?}", self.single_size, self.batch_size)
    }
}

impl<D: Device> DenseMatrix<D> {
    pub fn zeroed(device: Arc<D>, single_size: usize) -> Result<Self, D::DeviceError> {
        let buf = D::BufferF32::new(device, single_size)?;
        Ok(Self { buf, single_size, batch_size: None })
    }

    pub fn ones(device: Arc<D>, size: usize) -> Result<Self, D::DeviceError> {
        let mut res = Self { buf: D::BufferF32::new(device, size)?, single_size: size, batch_size: None };
        res.load_from_slice(None, &vec![1.0; size])?;
        Ok(res)
    }

    /// Calculates `self += scale * rhs`
    pub fn add(&mut self, scale: f32, rhs: &Self) -> Result<(), OperationError<D::DeviceError>> {
        if self.size() != rhs.size() {
            return Err(OperationError::IndexOutOfBounds);
        }

        self.buf.linear_comb(self.size(), 1.0, scale, &rhs.buf)?;

        Ok(())
    }

    /// Calculates `self = (1 - lambda) * self + lambda * rhs`
    pub fn lerp(&mut self, lambda: f32, rhs: &Self) -> Result<(), OperationError<D::DeviceError>> {
        if self.size() != rhs.size() {
            return Err(OperationError::IndexOutOfBounds);
        }

        self.buf.linear_comb(self.size(), 1.0 - lambda, lambda, &rhs.buf)?;

        Ok(())
    }

    pub fn clamp(&mut self, min: f32, max: f32) -> Result<(), OperationError<D::DeviceError>> {
        self.buf.clip(self.size(), min, max)?;
        Ok(())
    }

    pub fn scale(&mut self, scale: f32) -> Result<(), OperationError<D::DeviceError>> {
        self.buf.mul_scalar(self.size(), scale)?;
        Ok(())
    }

    pub fn allocated_size(&self) -> usize {
        self.buf.size()
    }

    pub fn size(&self) -> usize {
        self.single_size() * self.batch_size().unwrap_or(1)
    }

    pub fn single_size(&self) -> usize {
        self.single_size
    }

    pub fn swap_with(&mut self, other: &mut Self) -> Result<(), D::DeviceError> {
        if self.single_size != other.single_size {
            return Err(D::DeviceError::default());
        }

        std::mem::swap(self, other);

        Ok(())
    }

    pub fn copy_from(&mut self, other: &Self) -> Result<(), D::DeviceError> {
        if self.single_size != other.single_size {
            return Err(D::DeviceError::default());
        }

        self.set_batch_size(other.batch_size())?;
        self.buf.load_from_device(&other.buf, other.size())
    }

    pub fn batch_size(&self) -> Option<usize> {
        self.batch_size.map(NonZeroUsize::get)
    }

    pub fn set_batch_size(&mut self, batch_size: Option<usize>) -> Result<(), D::DeviceError> {
        let new_size = self.single_size() * batch_size.unwrap_or(1);
        if new_size > self.allocated_size() {
            self.buf = D::BufferF32::new(self.buf.device(), new_size)?;
        } else if batch_size != self.batch_size() {
            self.buf.set_zero()?;
        }

        self.batch_size = batch_size.map(|x| NonZeroUsize::new(x).unwrap());

        Ok(())
    }

    pub fn load_from_slice(&mut self, batch_size: Option<usize>, buf: &[f32]) -> Result<(), D::DeviceError> {
        if self.single_size() * batch_size.unwrap_or(1) != buf.len() {
            return Err(D::DeviceError::default());
        }

        self.set_batch_size(batch_size)?;
        self.buf.load_from_slice(buf)
    }

    /// # Safety
    /// Must synchronise before `buf` is dropped or mutated.
    pub unsafe fn load_non_blocking_from_host(
        &mut self,
        batch_size: Option<usize>,
        buf: &[f32],
    ) -> Result<(), D::DeviceError> {
        if self.single_size() * batch_size.unwrap_or(1) != buf.len() {
            return Err(D::DeviceError::default());
        }

        self.set_batch_size(batch_size)?;
        self.buf.load_non_blocking_from_host(buf)
    }

    pub fn set_to(&mut self, val: f32) -> Result<(), D::DeviceError> {
        self.buf.set_to(self.size(), val)
    }

    /// Writes the contents of this matrix into a buffer,
    /// returns number of values written.
    pub fn write_to_slice(&self, buf: &mut [f32]) -> Result<usize, D::DeviceError> {
        if self.size() > buf.len() {
            return Err(D::DeviceError::default());
        }

        self.buf.write_into_slice(buf, self.size())?;
        Ok(self.allocated_size())
    }

    /// Writes a complete description of the matrix into a buffer of bytes,
    /// along with an ID tag.
    pub fn write_to_byte_buffer(&self, id: &str) -> std::io::Result<Vec<u8>> {
        use std::io::{Error, ErrorKind, Write};

        if self.batch_size.is_some() {
            return Err(Error::new(ErrorKind::InvalidInput, "Cannot write batched!"));
        }

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
        buf.write_all(&usize::to_le_bytes(self.single_size))?;

        let mut values = vec![0.0; self.single_size];
        self.write_to_slice(&mut values).unwrap();

        for val in values {
            buf.write_all(&f32::to_le_bytes(val))?;
        }

        Ok(buf)
    }
}

/// Reads a matrix from a byte buffer, returning how many bytes were read
/// and the matrix ID that was read.
pub fn read_from_byte_buffer(bytes: &[u8], old_format: bool) -> (Vec<f32>, String, usize) {
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

    let mut single_size = [0u8; USIZE];
    single_size.copy_from_slice(&bytes[offset..offset + USIZE]);
    offset += USIZE;

    let mut single_size = usize::from_le_bytes(single_size);

    if old_format {
        let mut cols = [0u8; USIZE];
        cols.copy_from_slice(&bytes[offset..offset + USIZE]);
        offset += USIZE;
        single_size *= usize::from_le_bytes(cols);
    }

    let total_read = offset + single_size * 4;

    let mut values = vec![0.0; single_size];

    for (word, val) in bytes[offset..total_read].chunks_exact(4).zip(values.iter_mut()) {
        let mut buf = [0; 4];
        buf.copy_from_slice(word);
        *val = f32::from_le_bytes(buf);
    }

    (values, id, total_read)
}
