use std::sync::Arc;

use bullet_core::device::DeviceBuffer;

use cudarc::driver::{CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits};

use crate::ExecutionContext;

pub struct CudaBuffer<T> {
    pub(crate) buffer: CudaSlice<T>,
    pub(crate) device: Arc<ExecutionContext>,
}

impl<T: DeviceRepr + ValidAsZeroBits> DeviceBuffer<ExecutionContext, T> for CudaBuffer<T> {
    fn new(device: Arc<ExecutionContext>, size: usize) -> Self {
        Self { buffer: device.device.alloc_zeros(size).unwrap(), device }
    }

    fn device(&self) -> Arc<ExecutionContext> {
        self.device.clone()
    }

    fn size(&self) -> usize {
        self.buffer.len()
    }

    fn set_zero(&mut self) {
        self.buffer.device().memset_zeros(&mut self.buffer).unwrap();
    }

    fn load_from_device(&mut self, buf: &Self, bytes: usize) {
        let device = self.buffer.device();
        let src_slice = buf.buffer.slice(0..bytes);
        let mut dst_slice = self.buffer.slice_mut(0..bytes);
        device.dtod_copy(&src_slice, &mut dst_slice).unwrap();
    }

    fn load_from_slice(&mut self, buf: &[T]) {
        let device = self.buffer.device();
        let mut dst_slice = self.buffer.slice_mut(0..buf.len());
        device.htod_sync_copy_into(buf, &mut dst_slice).unwrap();
    }

    fn write_into_slice(&self, buf: &mut [T], bytes: usize) {
        let device = self.buffer.device();
        let src_slice = self.buffer.slice(0..bytes);
        device.dtoh_sync_copy_into(&src_slice, &mut buf[..bytes]).unwrap();
    }
}
