use std::sync::Arc;

use super::Device;

pub trait DeviceBuffer<D: Device, T>: Sized {
    fn new(device: Arc<D>, size: usize) -> Result<Self, D::DeviceError>;

    fn size(&self) -> usize;

    fn device(&self) -> Arc<D>;

    fn set_zero(&mut self) -> Result<(), D::DeviceError>;

    fn load_from_device(&mut self, buf: &Self, num: usize) -> Result<(), D::DeviceError>;

    fn load_from_slice(&mut self, buf: &[T]) -> Result<(), D::DeviceError>;

    fn write_into_slice(&self, buf: &mut [T], num: usize) -> Result<(), D::DeviceError>;
}
