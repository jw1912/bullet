use std::sync::Arc;

pub trait ValidType {}

impl ValidType for f32 {}
impl ValidType for i32 {}

pub trait Device: Sized {
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
