pub mod base;
pub mod blas;

use std::{fmt::Debug, sync::Arc};

use base::BaseOperations;
use blas::BlasOperations;

use crate::{
    cpu::CpuThread,
    graph::ir::{operation::unary::DiffableFromOutput, shape::Shape},
};

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

#[allow(clippy::too_many_arguments)]
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

    fn sparse_affine_activate(
        batch_size: usize,
        stride: Option<bool>,
        activation: DiffableFromOutput,
        input_a: &Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        input_b_vals: Option<&Self::BufferF32>,
        shape_b: Shape,
        nnz: usize,
        input_c: Option<&Self::BufferF32>,
        input_c_batched: bool,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn backprop_sparse_affine_activate(
        batch_size: usize,
        stride: Option<bool>,
        activation: DiffableFromOutput,
        input_a_grad: &mut Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        input_b_vals: Option<&Self::BufferF32>,
        shape_b: Shape,
        nnz: usize,
        input_c_grad: Option<&mut Self::BufferF32>,
        input_c_batched: bool,
        outputs: &Self::BufferF32,
        output_grad: &Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn select(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        input: &Self::BufferF32,
        indices: &Self::BufferI32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn select_backprop(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        indices: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn softmax_across_batch(
        batch_size: usize,
        single_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn crossentropy(
        size: usize,
        pred: &Self::BufferF32,
        target: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn backprop_softmax_crossentropy(
        size: usize,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn sparse_to_dense(
        batch_size: usize,
        size: usize,
        nnz: usize,
        sparse: &Self::BufferI32,
        dense: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;
}
