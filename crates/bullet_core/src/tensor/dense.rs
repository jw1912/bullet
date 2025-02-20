use std::{fmt::Debug, num::NonZeroUsize, sync::Arc};

use crate::device::{Device, DeviceBuffer};

pub struct DenseMatrix<D: Device> {
    pub(crate) buf: D::BufferF32,
    pub(crate) single_size: usize,
    pub(crate) batch_size: Option<NonZeroUsize>,
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

    pub fn allocated_size(&self) -> usize {
        self.buf.size()
    }

    pub fn size(&self) -> usize {
        self.single_size() * self.batch_size().unwrap_or(1)
    }

    pub fn single_size(&self) -> usize {
        self.single_size
    }

    pub fn copy_from(&mut self, other: &Self) -> Result<(), D::DeviceError> {
        assert_eq!(self.single_size, other.single_size);
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
        assert_eq!(self.single_size() * batch_size.unwrap_or(1), buf.len());
        self.set_batch_size(batch_size)?;
        self.buf.load_from_slice(buf)
    }

    pub fn set_zero(&mut self) -> Result<(), D::DeviceError> {
        self.buf.set_zero()
    }

    /// Writes the contents of this matrix into a buffer,
    /// returns number of values written.
    pub fn write_to_slice(&self, buf: &mut [f32]) -> Result<usize, D::DeviceError> {
        assert!(self.size() <= buf.len());
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
