mod sparse;

use std::sync::{Arc, Mutex};

use bullet_core::{
    device::{Device, DeviceBuffer, OperationError, OperationResult},
    graph::ir::{operation::unary::DiffableFromOutput, shape::Shape, BackendMarker},
};
use cudarc::{
    cublas::{result::CublasError, CudaBlas},
    driver::{CudaContext, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

use crate::CudaBuffer;

#[derive(Debug, Default)]
pub enum CudaError {
    #[default]
    Generic,
    Driver(DriverError),
    Blas(CublasError),
    ExpectedIllegalAddressAccess,
}

#[derive(Debug)]
pub struct CudaDevice {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    module: Arc<CudaModule>,
    copystream: Arc<CudaStream>,
    ones: Mutex<CudaSlice<f32>>,
}

impl Default for CudaDevice {
    fn default() -> Self {
        Self::new(0).unwrap()
    }
}

impl CudaDevice {
    pub fn stream(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    pub fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    pub fn module(&self) -> Arc<CudaModule> {
        self.module.clone()
    }

    pub fn copystream(&self) -> Arc<CudaStream> {
        self.copystream.clone()
    }

    pub fn with_ones<T, F: FnMut(&CudaSlice<f32>) -> Result<T, CudaError>>(
        self: Arc<Self>,
        count: usize,
        mut f: F,
    ) -> Result<T, CudaError> {
        let mut ones = self.ones.try_lock().unwrap();

        if count > ones.len() {
            *ones = self.stream.alloc_zeros(count).map_err(CudaError::Driver)?;

            crate::base::set_to(self.clone(), &mut ones, count, 1.0).map_err(CudaError::Driver)?;
        }

        f(&ones)
    }

    pub fn elementwise_launch_params(size: usize, threads: u32) -> LaunchConfig {
        let float4_size = (size as u32).div_ceil(4);
        let blocks = float4_size.div_ceil(threads);
        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
    }

    pub fn elementwise_launch_params_single(size: usize, threads: u32) -> LaunchConfig {
        let blocks = (size as u32).div_ceil(threads);
        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
    }
}

#[derive(Clone, Copy, Default)]
pub struct CudaMarker;
impl BackendMarker for CudaMarker {
    type Backend = CudaDevice;
}

#[allow(unused)]
impl Device for CudaDevice {
    type Marker = CudaMarker;
    type IdType = usize;
    type DeviceError = CudaError;
    type BufferF32 = CudaBuffer<f32>;
    type BufferI32 = CudaBuffer<i32>;

    fn new(id: Self::IdType) -> Result<Self, Self::DeviceError> {
        let ctx = CudaContext::new(id).map_err(CudaError::Driver)?;
        ctx.set_blocking_synchronize().map_err(CudaError::Driver)?;
        let stream = ctx.default_stream();
        let copystream = ctx.new_stream().map_err(CudaError::Driver)?;
        let blas = CudaBlas::new(stream.clone()).map_err(CudaError::Blas)?;

        static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

        let module = ctx.load_module(Ptx::from_src(PTX)).map_err(CudaError::Driver)?;

        let ones = Mutex::new(stream.alloc_zeros::<f32>(0).map_err(CudaError::Driver)?);

        Ok(Self { stream, blas, module, copystream, ones })
    }

    fn synchronise(&self) -> Result<(), Self::DeviceError> {
        self.stream.synchronize().map_err(CudaError::Driver)?;
        self.copystream.synchronize().map_err(CudaError::Driver)
    }

    fn get_last_device_error(&self) -> Result<(), Self::DeviceError> {
        self.synchronise()
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
    ) -> OperationResult<Self::DeviceError> {
        if input_b_vals.is_some() {
            return Err(OperationError::UnsupportedOperation);
        }

        sparse::sparse_affine(
            batch_size,
            stride,
            activation,
            input_a,
            shape_a,
            input_b,
            shape_b,
            nnz,
            input_c,
            input_c_batched,
            output,
        )
    }

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
    ) -> OperationResult<Self::DeviceError> {
        if input_b_vals.is_some() {
            return Err(OperationError::UnsupportedOperation);
        }

        sparse::backprop_sparse_affine(
            batch_size,
            stride,
            activation,
            input_a_grad,
            shape_a,
            input_b,
            shape_b,
            nnz,
            input_c_grad,
            input_c_batched,
            outputs,
            output_grad,
        )
    }

    fn select(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        input: &Self::BufferF32,
        indices: &Self::BufferI32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        if batch_size * input_size > input.size()
            || batch_size > indices.size()
            || batch_size * output_size > output.size()
        {
            return OperationResult::Err(OperationError::IndexOutOfBounds);
        }

        let func = input.device.module.load_function("select").map_err(CudaError::Driver)?;

        let threads = 1024;
        let grid_dim = (((batch_size * output_size) as u32).div_ceil(threads), 1, 1);
        let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            input
                .device
                .stream
                .launch_builder(&func)
                .arg(&(batch_size as i32))
                .arg(&(input_size as i32))
                .arg(&(output_size as i32))
                .arg(&indices.buf)
                .arg(&input.buf)
                .arg(&mut output.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn select_backprop(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        indices: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        if batch_size * input_size > input_grad.size()
            || batch_size > indices.size()
            || batch_size * output_size > output_grad.size()
        {
            return OperationResult::Err(OperationError::IndexOutOfBounds);
        }

        let func = input_grad.device.module.load_function("select_backprop").map_err(CudaError::Driver)?;

        let threads = 1024;
        let grid_dim = (((batch_size * output_size) as u32).div_ceil(threads), 1, 1);
        let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            input_grad
                .device
                .stream
                .launch_builder(&func)
                .arg(&(batch_size as i32))
                .arg(&(input_size as i32))
                .arg(&(output_size as i32))
                .arg(&indices.buf)
                .arg(&output_grad.buf)
                .arg(&mut input_grad.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn softmax_across_batch(
        batch_size: usize,
        single_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn crossentropy(
        size: usize,
        pred: &Self::BufferF32,
        target: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn backprop_softmax_crossentropy(
        size: usize,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn sparse_to_dense(
        batch_size: usize,
        size: usize,
        nnz: usize,
        sparse: &Self::BufferI32,
        dense: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }
}
