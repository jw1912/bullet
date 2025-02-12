use std::sync::Arc;

use super::Device;

pub trait DeviceBuffer<D: Device, T> {
    fn new(device: Arc<D>, size: usize) -> Self;

    fn size(&self) -> usize;

    fn device(&self) -> Arc<D>;

    fn set_zero(&mut self);

    fn load_from_device(&mut self, buf: &Self, bytes: usize);

    fn load_from_slice(&mut self, buf: &[T]);

    fn write_into_slice(&self, buf: &mut [T], bytes: usize);
}
