use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

#[cfg(feature = "cuda")]
/// # Safety
/// Must be representable on all devices and valid
/// as zeroed bits
pub unsafe trait ValidType: DeviceRepr + ValidAsZeroBits {}

#[cfg(not(feature = "cuda"))]
/// # Safety
/// Must be representable on all devices and valid
/// as zeroed bits
pub unsafe trait ValidType {}

unsafe impl ValidType for f32 {}
unsafe impl ValidType for i32 {}

pub trait Device: Sized + 'static {
    type IdType;
    type Buffer<T: ValidType>: DeviceBuffer<Self, T>;

    fn new(id: Self::IdType) -> Self;

    fn synchronise(&self);

    fn panic_if_device_error(&self, msg: &str);
}

pub trait DeviceBuffer<D: Device, T: ValidType> {
    fn new(device: Arc<D>, size: usize) -> Self;

    fn size(&self) -> usize;

    fn device(&self) -> Arc<D>;

    fn set_zero(&mut self);

    fn load_from_device(&mut self, buf: &Self, bytes: usize);

    fn load_from_slice(&mut self, buf: &[T]);

    fn write_into_slice(&self, buf: &mut [T], bytes: usize);
}
