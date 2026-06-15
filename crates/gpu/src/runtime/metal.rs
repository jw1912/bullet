//! Minimal wrapper around the Metal runtime for Apple Silicon GPUs

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    ffi::{CStr, c_char, c_int, c_uint, c_void},
};

use objc2::AllocAnyThread;
use objc2::ffi::NSUInteger;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication};

use super::bindings::{Dim3, GpuBindings};
use crate::runtime::Dialect;
use crate::runtime::bindings::{DeviceProps, GemmConfig};

thread_local! {
    static CURRENT_DEVICE: Cell<i32> = const { Cell::new(0) };
    static NEXT_ID: Cell<u64> = const { Cell::new(1) };
    static BUFFER_REGISTRY:   RefCell<HashMap<u64, Retained<ProtocolObject<dyn MTLBuffer>>>>               = RefCell::new(HashMap::new());
    static LIBRARY_REGISTRY:  RefCell<HashMap<u64, Retained<ProtocolObject<dyn MTLLibrary>>>>              = RefCell::new(HashMap::new());
    static DEVICE_REGISTRY:   RefCell<HashMap<i32, Retained<ProtocolObject<dyn MTLDevice>>>>               = RefCell::new(HashMap::new());
    static QUEUE_REGISTRY:    RefCell<HashMap<u64, Retained<ProtocolObject<dyn MTLCommandQueue>>>>         = RefCell::new(HashMap::new());
    static PIPELINE_REGISTRY: RefCell<HashMap<u64, Retained<ProtocolObject<dyn MTLComputePipelineState>>>> = RefCell::new(HashMap::new());
    static BLAS_STREAM_MAP:   RefCell<HashMap<u64, u64>>                                                   = RefCell::new(HashMap::new());
}

fn next_id() -> u64 {
    NEXT_ID.with(|n| {
        let id = n.get();
        n.set(id + 1);
        id
    })
}

/// Returns a raw pointer to the current Metal device.
///
/// ### SAFETY
///
/// Devices are inserted once and never removed, so the
/// pointer remains valid for the lifetime of the program
unsafe fn current_device() -> *const ProtocolObject<dyn MTLDevice> {
    let ordinal = CURRENT_DEVICE.with(|c| c.get());
    DEVICE_REGISTRY.with_borrow(|r| Retained::as_ptr(r.get(&ordinal).expect("Metal device not initialised")))
}

/// Marker for the Metal runtime
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Metal;

/// Error type for the Metal runtime
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalError {
    Runtime(String),
    Compile(String),
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Runtime(s) | Self::Compile(s) => write!(f, "{s}"),
        }
    }
}

impl From<String> for MetalError {
    fn from(value: String) -> Self {
        Self::Runtime(value)
    }
}

type MetalResult = Result<(), MetalError>;

#[allow(unsafe_op_in_unsafe_fn)]
impl GpuBindings for Metal {
    type Err = MetalError;
    type Dev = c_int;
    type Ptr = u64;
    type Ctx = c_int;
    type Stream = u64;
    type BlasHandle = u64;
    type Kernel = u64;
    type Module = u64;

    unsafe fn driver_init() -> MetalResult {
        Ok(())
    }

    unsafe fn device_get(ordinal: c_int) -> Result<c_int, MetalError> {
        let device =
            MTLCreateSystemDefaultDevice().ok_or_else(|| MetalError::Runtime("No Metal device found".into()))?;
        DEVICE_REGISTRY.with_borrow_mut(|r| r.insert(ordinal, device));
        Ok(ordinal)
    }

    unsafe fn device_props(device: c_int) -> Result<DeviceProps, MetalError> {
        DEVICE_REGISTRY.with_borrow(|r| {
            let dev = r.get(&device).ok_or_else(|| MetalError::Runtime("Device not found".into()))?;
            Ok(DeviceProps {
                name: dev.name().to_string(),
                warp_size: Some(32),
                stream_mem_alloc: false,
                vec_atomics: true,
                arch: None,
                dialect: Dialect::Msl,
            })
        })
    }

    unsafe fn context_create(device: c_int) -> Result<c_int, MetalError> {
        CURRENT_DEVICE.with(|c| c.set(device));
        Ok(device)
    }

    unsafe fn context_destroy(_device: c_int) -> MetalResult {
        Ok(())
    }

    unsafe fn context_set(ctx: c_int) -> MetalResult {
        CURRENT_DEVICE.with(|c| c.set(ctx));
        Ok(())
    }

    unsafe fn context_sync() -> MetalResult {
        Ok(())
    }

    unsafe fn context_malloc(bytes: usize) -> Result<u64, MetalError> {
        let device = &*current_device();
        let buffer = device
            .newBufferWithLength_options(bytes as NSUInteger, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::Runtime("Failed to allocate Metal buffer".into()))?;
        let id = next_id();
        BUFFER_REGISTRY.with_borrow_mut(|r| r.insert(id, buffer));
        Ok(id)
    }

    unsafe fn context_free(dev_ptr: u64) -> MetalResult {
        BUFFER_REGISTRY
            .with_borrow_mut(|r| r.remove(&dev_ptr))
            .ok_or_else(|| MetalError::Runtime(format!("Buffer {dev_ptr} not found")))?;
        Ok(())
    }

    unsafe fn context_memset(dev_ptr: u64, bytes: usize, value: u8) -> MetalResult {
        let ptr = BUFFER_REGISTRY.with_borrow(|r| {
            let entry = r.get(&dev_ptr).ok_or_else(|| MetalError::Runtime(format!("Buffer {dev_ptr} not found")))?;
            Ok::<_, MetalError>(entry.contents().as_ptr() as *mut u8)
        })?;
        std::ptr::write_bytes(ptr, value, bytes);
        Ok(())
    }

    unsafe fn context_memcpy_d2h(dst: *mut c_void, src: u64, bytes: usize) -> MetalResult {
        let src_ptr = BUFFER_REGISTRY.with_borrow(|r| {
            let entry = r.get(&src).ok_or_else(|| MetalError::Runtime(format!("Buffer {src} not found")))?;
            Ok::<_, MetalError>(entry.contents().as_ptr() as *const u8)
        })?;
        std::ptr::copy_nonoverlapping(src_ptr, dst as *mut u8, bytes);
        Ok(())
    }

    unsafe fn context_memcpy_h2d(dst: u64, src: *const c_void, bytes: usize) -> MetalResult {
        let dst_ptr = BUFFER_REGISTRY.with_borrow(|r| {
            let entry = r.get(&dst).ok_or_else(|| MetalError::Runtime(format!("Buffer {dst} not found")))?;
            Ok::<_, MetalError>(entry.contents().as_ptr())
        })?;
        std::ptr::copy_nonoverlapping(src as *const u8, dst_ptr as *mut u8, bytes);
        Ok(())
    }

    unsafe fn stream_create() -> Result<u64, MetalError> {
        let device = &*current_device();
        let queue =
            device.newCommandQueue().ok_or_else(|| MetalError::Runtime("Failed to create command queue".into()))?;
        let id = next_id();
        QUEUE_REGISTRY.with_borrow_mut(|r| r.insert(id, queue));
        Ok(id)
    }

    unsafe fn stream_destroy(stream: u64) -> MetalResult {
        QUEUE_REGISTRY.with_borrow_mut(|r| r.remove(&stream));
        Ok(())
    }

    unsafe fn stream_sync(stream: u64) -> MetalResult {
        autoreleasepool(|_| {
            let command_buffer = QUEUE_REGISTRY.with_borrow(|r| {
                let queue = r.get(&stream).ok_or_else(|| MetalError::Runtime("Stream not found".into()))?;
                queue.commandBuffer().ok_or_else(|| MetalError::Runtime("Failed to create command buffer".into()))
            })?;
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            Ok(())
        })
    }

    unsafe fn stream_malloc(_stream: u64, bytes: usize) -> Result<u64, MetalError> {
        Self::context_malloc(bytes)
    }

    unsafe fn stream_free(_stream: u64, dev_ptr: u64) -> MetalResult {
        Self::context_free(dev_ptr)
    }

    unsafe fn stream_memset(_stream: u64, dev_ptr: u64, bytes: usize, value: u8) -> MetalResult {
        Self::context_memset(dev_ptr, bytes, value)
    }

    unsafe fn stream_memcpy_d2h(_stream: u64, dst: *mut c_void, src: u64, bytes: usize) -> MetalResult {
        Self::context_memcpy_d2h(dst, src, bytes)
    }

    unsafe fn stream_memcpy_h2d(_stream: u64, dst: u64, src: *const c_void, bytes: usize) -> MetalResult {
        Self::context_memcpy_h2d(dst, src, bytes)
    }

    unsafe fn kernel_load(_kernel: u64) -> MetalResult {
        Ok(())
    }

    unsafe fn kernel_destroy(kernel: u64) -> MetalResult {
        PIPELINE_REGISTRY.with_borrow_mut(|r| r.remove(&kernel));
        Ok(())
    }

    unsafe fn kernel_launch(
        func: u64,
        stream: u64,
        grid_dim: Dim3,
        block_dim: Dim3,
        args: &mut [*mut c_void],
        _smem: c_uint,
    ) -> MetalResult {
        autoreleasepool(|_| unsafe {
            let command_buffer = QUEUE_REGISTRY.with_borrow(|r| {
                let queue = r.get(&stream).ok_or_else(|| MetalError::Runtime("Stream not found".into()))?;
                queue.commandBuffer().ok_or_else(|| MetalError::Runtime("Failed to create command buffer".into()))
            })?;

            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| MetalError::Runtime("Failed to create compute encoder".into()))?;

            PIPELINE_REGISTRY.with_borrow(|r| {
                let pipeline = r.get(&func).ok_or_else(|| MetalError::Runtime("Kernel not found".into()))?;
                encoder.setComputePipelineState(pipeline);
                Ok::<(), MetalError>(())
            })?;

            let total_threads = grid_dim.x * block_dim.x;

            BUFFER_REGISTRY.with_borrow(|buf_reg| {
                for (index, &mut arg_ptr) in args.iter_mut().enumerate() {
                    let buffer_id = *(arg_ptr as *const u64);
                    let entry = buf_reg
                        .get(&buffer_id)
                        .ok_or_else(|| MetalError::Runtime(format!("Buffer {buffer_id} not found for arg {index}")))?;
                    encoder.setBuffer_offset_atIndex(Some(&*entry), 0, index as NSUInteger);
                }
                Ok::<(), MetalError>(())
            })?;

            let threads_per_grid = MTLSize {
                width: (total_threads) as NSUInteger,
                height: grid_dim.y as NSUInteger,
                depth: grid_dim.z as NSUInteger,
            };
            let threads_per_threadgroup = MTLSize {
                width: block_dim.x as NSUInteger,
                height: block_dim.y as NSUInteger,
                depth: block_dim.z as NSUInteger,
            };

            encoder.dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup);
            encoder.endEncoding();
            command_buffer.commit();

            Ok(())
        })
    }

    unsafe fn module_create(code: *const c_void) -> Result<u64, MetalError> {
        Ok(*(code as *const u64))
    }

    unsafe fn module_destroy(module: u64) -> MetalResult {
        LIBRARY_REGISTRY.with_borrow_mut(|r| r.remove(&module));
        Ok(())
    }

    unsafe fn module_get_kernel(module: u64, kernel_name: &CStr) -> Result<u64, MetalError> {
        let name = kernel_name.to_str().map_err(|e| MetalError::Runtime(format!("{e}")))?;
        let ns_name = NSString::from_str(name);

        let function = LIBRARY_REGISTRY.with_borrow(|r| {
            let library = r.get(&module).ok_or_else(|| MetalError::Runtime("Module/library not found".into()))?;
            library
                .newFunctionWithName(&ns_name)
                .ok_or_else(|| MetalError::Runtime(format!("Function '{name}' not found in library")))
        })?;

        let device = &*current_device();
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::Compile(format!("Failed to create pipeline: {}", e.localizedDescription())))?;

        let id = next_id();
        PIPELINE_REGISTRY.with_borrow_mut(|r| r.insert(id, pipeline));
        Ok(id)
    }

    unsafe fn program_compile(
        source_code: &CStr,
        _num_options: c_int,
        _options: *const *const c_char,
    ) -> Result<Vec<c_char>, MetalError> {
        let source = source_code.to_str().map_err(|e| MetalError::Compile(format!("Invalid source: {e}")))?;

        let device = &*current_device();
        let options = MTLCompileOptions::new();
        let ns_source = NSString::from_str(source);
        let library = device
            .newLibraryWithSource_options_error(&ns_source, Some(&options))
            .map_err(|e| MetalError::Compile(format!("MSL compilation failed: {}", e.localizedDescription())))?;

        let id = next_id();
        LIBRARY_REGISTRY.with_borrow_mut(|r| r.insert(id, library));

        let id_bytes = id.to_ne_bytes();
        let result: Vec<c_char> = id_bytes.iter().map(|&b| b as c_char).collect();
        Ok(result)
    }

    unsafe fn blas_create() -> Result<u64, MetalError> {
        Ok(next_id())
    }

    unsafe fn blas_destroy(handle: u64) -> MetalResult {
        BLAS_STREAM_MAP.with_borrow_mut(|r| r.remove(&handle));
        Ok(())
    }

    unsafe fn blas_set_stream(handle: u64, stream: u64) -> MetalResult {
        BLAS_STREAM_MAP.with_borrow_mut(|r| r.insert(handle, stream));
        Ok(())
    }

    unsafe fn blas_gemm(handle: u64, config: GemmConfig, a: u64, b: u64, c: u64) -> MetalResult {
        Self::blas_gemm_batched(handle, 1, config, a, b, c)
    }

    unsafe fn blas_gemm_batched(
        handle: u64,
        batch_size: c_int,
        config: GemmConfig,
        a: u64,
        b: u64,
        c: u64,
    ) -> MetalResult {
        autoreleasepool(|_| unsafe {
            let stream_id = BLAS_STREAM_MAP.with_borrow(|r| {
                r.get(&handle)
                    .copied()
                    .ok_or_else(|| MetalError::Runtime("BLAS handle has no associated stream".into()))
            })?;

            let command_buffer = QUEUE_REGISTRY.with_borrow(|r| {
                let queue = r.get(&stream_id).ok_or_else(|| MetalError::Runtime("Stream not found for BLAS".into()))?;
                queue.commandBuffer().ok_or_else(|| MetalError::Runtime("Failed to create command buffer".into()))
            })?;

            let m = config.m as u64;
            let n = config.n as u64;
            let k = config.k as u64;
            let batch = batch_size as u64;

            // MPS is row-major
            let lda = if config.row_mjr_a { k } else { m };
            let ldb = if config.row_mjr_b { n } else { k };
            let ldc = m;

            let stride_a = m * k;
            let stride_b = k * n;
            let stride_c = m * n;

            let (a_rows, a_cols) = if config.row_mjr_a { (m, k) } else { (k, m) };
            let (b_rows, b_cols) = if config.row_mjr_b { (k, n) } else { (n, k) };

            let device = &*current_device();

            let (mat_a, mat_b, mat_c) = BUFFER_REGISTRY.with_borrow(|buf_reg| {
                let buf_a = &buf_reg.get(&a).ok_or_else(|| MetalError::Runtime("Buffer A not found".into()))?;
                let buf_b = &buf_reg.get(&b).ok_or_else(|| MetalError::Runtime("Buffer B not found".into()))?;
                let buf_c = &buf_reg.get(&c).ok_or_else(|| MetalError::Runtime("Buffer C not found".into()))?;

                let desc_a = mps_matrix_descriptor(a_rows, a_cols, lda, batch, stride_a);
                let desc_b = mps_matrix_descriptor(b_rows, b_cols, ldb, batch, stride_b);
                let desc_c = mps_matrix_descriptor(n, m, ldc, batch, stride_c);

                let mat_a = mps_matrix(buf_a, &desc_a);
                let mat_b = mps_matrix(buf_b, &desc_b);
                let mat_c = mps_matrix(buf_c, &desc_c);
                Ok::<_, MetalError>((mat_a, mat_b, mat_c))
            })?;

            let gemm = MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                device,
                config.row_mjr_b,
                config.row_mjr_a,
                n as NSUInteger,
                m as NSUInteger,
                k as NSUInteger,
                config.alpha as f64,
                config.beta as f64,
            );

            if batch > 1 {
                gemm.setBatchStart(0);
                gemm.setBatchSize(batch as NSUInteger);
            }

            gemm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(&command_buffer, &mat_b, &mat_a, &mat_c);
            command_buffer.commit();

            Ok(())
        })
    }
}

fn mps_matrix_descriptor(
    rows: u64,
    columns: u64,
    row_numel: u64,
    matrices: u64,
    matrix_numel: u64,
) -> Retained<MPSMatrixDescriptor> {
    unsafe {
        if matrices > 1 {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType(
                rows as NSUInteger,
                columns as NSUInteger,
                matrices as NSUInteger,
                (row_numel * 4) as NSUInteger,
                (matrix_numel * 4) as NSUInteger,
                MPSDataType::Float32,
            )
        } else {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                rows as NSUInteger,
                columns as NSUInteger,
                (row_numel * 4) as NSUInteger,
                MPSDataType::Float32,
            )
        }
    }
}

fn mps_matrix(buffer: &ProtocolObject<dyn MTLBuffer>, descriptor: &MPSMatrixDescriptor) -> Retained<MPSMatrix> {
    unsafe { MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), buffer, descriptor) }
}
