mod sparse_bwd;
mod sparse_fwd;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use bullet_core::{
    device::{Device, DeviceBuffer, OperationError, OperationResult},
    graph::ir::{operation::unary::DiffableFromOutput, shape::Shape, BackendMarker},
};
use cudarc::{
    cublas::{result::CublasError, CudaBlas},
    driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::{self, CompileError},
};

use crate::CudaBuffer;

#[derive(Debug, Default)]
pub enum CudaError {
    #[default]
    Generic,
    RuntimeCompile(CompileError),
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
    rtc: Mutex<HashMap<String, Arc<CudaModule>>>,
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

    /// # Safety
    /// Function name collisions can cause UB.
    pub unsafe fn get_custom_func_or_rtc<F: FnMut() -> String>(
        &self,
        name: &str,
        mut f: F,
    ) -> Result<CudaFunction, CudaError> {
        let mut rtcs = self.rtc.try_lock().unwrap();

        let module = if let Some(module) = rtcs.get(name) {
            module.clone()
        } else {
            let kernel_str = f();
            let ptx = nvrtc::compile_ptx(kernel_str).map_err(CudaError::RuntimeCompile)?;
            let module = self.stream.context().load_module(ptx).map_err(CudaError::Driver)?;
            rtcs.insert(name.to_string(), module.clone());
            module
        };

        module.load_function("kernel").map_err(CudaError::Driver)
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

        static KERNELS: &str = include_str!("kernels.cu");
        let ptx = nvrtc::compile_ptx(KERNELS).map_err(CudaError::RuntimeCompile)?;

        let module = ctx.load_module(ptx).map_err(CudaError::Driver)?;

        let ones = Mutex::new(stream.alloc_zeros::<f32>(0).map_err(CudaError::Driver)?);

        Ok(Self { stream, blas, module, copystream, ones, rtc: Mutex::new(HashMap::new()) })
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

        sparse_fwd::sparse_affine(
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

        sparse_bwd::backprop_sparse_affine(
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
        let size = batch_size * single_size;
        if size > input.size() || size > output.size() {
            return Err(OperationError::IndexOutOfBounds);
        }

        let func = input.device.module.load_function("softmax").unwrap();

        let threads = 512u32;
        let grid_dim = ((batch_size as u32).div_ceil(threads), 1, 1);
        let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            input
                .device
                .stream
                .launch_builder(&func)
                .arg(&(single_size as i32))
                .arg(&(batch_size as i32))
                .arg(&input.buf.slice(0..size))
                .arg(&mut output.buf.slice_mut(0..size))
                .launch(cfg);
        }

        Ok(())
    }

    fn crossentropy(
        size: usize,
        pred: &Self::BufferF32,
        target: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        if size > pred.size() || size > target.size() || size > output.size() {
            return Err(OperationError::IndexOutOfBounds);
        }

        let func = pred.device.module.load_function("cross_entropy").unwrap();
        let threads = 512u32;
        let grid_dim = ((size as u32).div_ceil(threads), 1, 1);
        let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            pred.device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&pred.buf.slice(0..size))
                .arg(&target.buf.slice(0..size))
                .arg(&mut output.buf.slice_mut(0..size))
                .launch(cfg);
        }

        Ok(())
    }

    fn backprop_softmax_crossentropy(
        size: usize,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        if size > softmaxed.size() || size > target.size() || size > output_grad.size() || size > input_grad.size() {
            return Err(OperationError::IndexOutOfBounds);
        }

        let func = softmaxed.device.module.load_function("backprop_softmax_cross_entropy").unwrap();
        let threads = 512u32;
        let grid_dim = ((size as u32).div_ceil(threads), 1, 1);
        let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            softmaxed
                .device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&softmaxed.buf.slice(0..size))
                .arg(&target.buf.slice(0..size))
                .arg(&output_grad.buf.slice(0..size))
                .arg(&mut input_grad.buf.slice_mut(0..size))
                .launch(cfg);
        }

        Ok(())
    }

    fn sparse_to_dense(
        batch_size: usize,
        size: usize,
        nnz: usize,
        sparse: &Self::BufferI32,
        dense: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        if batch_size * nnz > sparse.size() || batch_size * size > dense.size() {
            return Err(OperationError::IndexOutOfBounds);
        }

        dense.set_zero()?;

        let func = sparse.device.module.load_function("sparse_to_dense").unwrap();
        let threads = batch_size.min(1024);
        let blocks = batch_size.div_ceil(threads) as u32;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            sparse
                .device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&(batch_size as i32))
                .arg(&(nnz as i32))
                .arg(&sparse.buf)
                .arg(&mut dense.buf)
                .launch(cfg);
        }

        Ok(())
    }
}
