pub mod base;
pub mod blas;
pub mod cmp;
pub mod multi;
pub mod sparse;

use std::sync::Arc;

use crate::{
    device::{Device, DeviceBuffer, OperationError, OperationResult, operation::CoreDeviceOps},
    graph::ir::BackendMarker,
};

#[derive(Debug, Default)]
pub struct CpuError;

#[derive(Debug, Default)]
pub struct CpuThread;

pub struct CpuBuffer<T> {
    buf: Vec<T>,
    device: Arc<CpuThread>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuMarker;

impl BackendMarker for CpuMarker {
    type Backend = CpuThread;
}

impl<T: Copy + Default + Send + Sync> DeviceBuffer<CpuThread, T> for CpuBuffer<T> {
    type BufferError = CpuError;

    fn device(&self) -> Arc<CpuThread> {
        self.device.clone()
    }

    fn new(device: Arc<CpuThread>, size: usize) -> Result<Self, CpuError> {
        Ok(Self { buf: vec![T::default(); size], device })
    }

    fn size(&self) -> usize {
        self.buf.len()
    }

    fn set_zero(&mut self) -> Result<(), CpuError> {
        for elem in &mut self.buf {
            *elem = T::default();
        }

        Ok(())
    }

    fn load_from_device(&mut self, buf: &Self, num: usize) -> Result<(), CpuError> {
        self.buf[..num].copy_from_slice(&buf.buf[..num]);
        Ok(())
    }

    fn load_from_slice(&mut self, buf: &[T]) -> Result<(), CpuError> {
        self.buf[..buf.len()].copy_from_slice(buf);
        Ok(())
    }

    unsafe fn load_non_blocking_from_host(&mut self, buf: &[T]) -> Result<(), Self::BufferError> {
        self.load_from_slice(buf)
    }

    fn write_into_slice(&self, buf: &mut [T], num: usize) -> Result<(), CpuError> {
        buf[..num].copy_from_slice(&self.buf[..num]);
        Ok(())
    }
}

impl Device for CpuThread {
    type Marker = CpuMarker;

    type BufferF32 = CpuBuffer<f32>;
    type BufferI32 = CpuBuffer<i32>;

    type DeviceError = CpuError;

    type IdType = ();

    fn new(_id: Self::IdType) -> Result<Self, Self::DeviceError> {
        Ok(Self)
    }

    fn synchronise(&self) -> Result<(), Self::DeviceError> {
        Ok(())
    }

    fn get_last_device_error(&self) -> Result<(), Self::DeviceError> {
        Ok(())
    }

    fn sparse_to_dense(
        _batch_size: usize,
        _size: usize,
        _nnz: usize,
        _sparse: &Self::BufferI32,
        _dense: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }
}

impl CoreDeviceOps for CpuThread {
    fn select(
        batch_size: usize,
        input_batched: bool,
        input_size: usize,
        output_size: usize,
        input: &Self::BufferF32,
        indices: &Self::BufferI32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        for i in 0..batch_size {
            let bucket = indices.buf[i] as usize;
            let offset = if input_batched { i * input_size } else { 0 };

            for j in 0..output_size {
                output.buf[output_size * i + j] = input.buf[offset + output_size * bucket + j];
            }
        }

        Ok(())
    }

    fn select_backprop(
        batch_size: usize,
        input_batched: bool,
        input_size: usize,
        output_size: usize,
        indices: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        for i in 0..batch_size {
            let bucket = indices.buf[i] as usize;
            let offset = if input_batched { i * input_size } else { 0 };

            for j in 0..output_size {
                input_grad.buf[offset + output_size * bucket + j] += output_grad.buf[output_size * i + j];
            }
        }

        Ok(())
    }

    #[allow(unused)]
    fn softmax_across_batch(
        batch_size: usize,
        single_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    #[allow(unused)]
    fn crossentropy(
        size: usize,
        pred: &Self::BufferF32,
        target: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    #[allow(unused)]
    fn backprop_softmax_crossentropy(
        size: usize,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }
}
