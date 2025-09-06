pub mod base;
pub mod blas;
pub mod core;
pub mod sparse;

use std::{fmt::Debug, sync::Arc};

pub use base::BaseOperations;
pub use blas::BlasOperations;
pub use core::CoreDeviceOps;
pub use sparse::SparseAffineOps;

use crate::cpu::CpuThread;

#[derive(Debug)]
pub enum OperationError<T: Debug> {
    TensorOptimisedOut,
    InvalidTensorFormat,
    IndexOutOfBounds,
    UnsupportedOperation,
    MismatchedBatchSizes,
    NoWeightWithID(String),
    WeightLoadingError(String, Option<(usize, usize)>),
    DeviceError(Box<T>),
}

impl<T: Debug> From<T> for OperationError<T> {
    fn from(value: T) -> Self {
        Self::DeviceError(Box::new(value))
    }
}

pub type OperationResult<T> = Result<(), OperationError<T>>;

pub trait DeviceBuffer<D, T>: Sized {
    type BufferError;

    fn new(device: Arc<D>, size: usize) -> Result<Self, Self::BufferError>;

    fn size(&self) -> usize;

    fn device(&self) -> Arc<D>;

    fn set_zero(&mut self) -> Result<(), Self::BufferError>;

    fn load_from_device(&mut self, buf: &Self, num: usize) -> Result<(), Self::BufferError>;

    fn load_from_slice(&mut self, buf: &[T]) -> Result<(), Self::BufferError>;

    /// # Safety
    /// Needs to be followed by a synchronise before `buf` is dropped!
    unsafe fn load_non_blocking_from_host(&mut self, buf: &[T]) -> Result<(), Self::BufferError>;

    fn write_into_slice(&self, buf: &mut [T], num: usize) -> Result<(), Self::BufferError>;
}

pub trait Device: Sized + 'static {
    type IdType;
    type Marker;
    type DeviceError: std::fmt::Debug + Default;
    type BufferI32: DeviceBuffer<Self, i32, BufferError = Self::DeviceError>;
    type BufferF32: DeviceBuffer<Self, f32, BufferError = Self::DeviceError>
        + BaseOperations<BaseError = Self::DeviceError>
        + BlasOperations<BlasError = Self::DeviceError>;

    fn new(id: Self::IdType) -> Result<Self, Self::DeviceError>;

    fn synchronise(&self) -> Result<(), Self::DeviceError>;

    fn get_last_device_error(&self) -> Result<(), Self::DeviceError>;

    fn sanity_check(self: Arc<Self>) {
        println!("\x1b[34;1mRunning Sanity Checks\x1b[0m");
        CpuThread::compare_linear_comb(self.clone());
        CpuThread::compare_gemm(self.clone());
        CpuThread::compare_gebmm(self.clone());
        CpuThread::compare_activate(self.clone());
        CpuThread::compare_power_error(self.clone());
        CpuThread::compare_pairwise(self.clone());
        CpuThread::compare_clip(self.clone());
        CpuThread::compare_adam(self.clone());
        CpuThread::compare_copy_or_add_strided(self.clone());
        CpuThread::compare_add(self.clone());
        CpuThread::compare_abs_pow(self.clone());
    }

    fn sparse_to_dense(
        batch_size: usize,
        size: usize,
        nnz: usize,
        sparse: &Self::BufferI32,
        dense: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;
}
