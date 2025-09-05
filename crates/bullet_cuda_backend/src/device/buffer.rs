use std::sync::Arc;

use bullet_core::device::DeviceBuffer;
use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};

use crate::{CudaDevice, CudaError};

#[derive(Debug)]
pub struct CudaBuffer<T> {
    pub buf: CudaSlice<T>,
    pub device: Arc<CudaDevice>,
}

impl<T: DeviceRepr + ValidAsZeroBits> DeviceBuffer<CudaDevice, T> for CudaBuffer<T> {
    type BufferError = CudaError;

    fn new(device: Arc<CudaDevice>, size: usize) -> Result<Self, Self::BufferError> {
        device
            .stream()
            .alloc_zeros::<T>(size)
            .map(|ok| Self { buf: ok, device: device.clone() })
            .map_err(CudaError::Driver)
    }

    fn device(&self) -> Arc<CudaDevice> {
        self.device.clone()
    }

    fn size(&self) -> usize {
        self.buf.len()
    }

    fn set_zero(&mut self) -> Result<(), Self::BufferError> {
        self.device.stream().memset_zeros(&mut self.buf).map_err(CudaError::Driver)
    }

    fn load_from_device(&mut self, buf: &Self, num: usize) -> Result<(), Self::BufferError> {
        self.device
            .stream()
            .memcpy_dtod(&buf.buf.slice(0..num), &mut self.buf.slice_mut(0..num))
            .map_err(CudaError::Driver)
    }

    fn load_from_slice(&mut self, buf: &[T]) -> Result<(), Self::BufferError> {
        self.device.stream().memcpy_htod(buf, &mut self.buf.slice_mut(0..buf.len())).map_err(CudaError::Driver)
    }

    unsafe fn load_non_blocking_from_host(&mut self, buf: &[T]) -> Result<(), Self::BufferError> {
        self.device.copystream().memcpy_htod(buf, &mut self.buf.slice_mut(0..buf.len())).map_err(CudaError::Driver)
    }

    fn write_into_slice(&self, buf: &mut [T], num: usize) -> Result<(), Self::BufferError> {
        self.device.stream().memcpy_dtoh(&self.buf.slice(0..num), &mut buf[..num]).map_err(CudaError::Driver)
    }
}
