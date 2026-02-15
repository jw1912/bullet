//! Minimal runtime around CUDA/ROCm devices

mod bindings;
#[cfg(feature = "cuda")]
pub mod cuda;

use std::{
    ffi::{CString, c_void},
    fmt,
    hash::Hash,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

pub use bindings::Dim3;

/// Marker trait for the CUDA and ROCm runtimes to implement
pub trait Gpu: bindings::GpuBindings<Err = Self::Error, Ptr = Self::DevicePtr> {
    type Error: fmt::Debug + Eq + From<String>;
    type DevicePtr: Copy + Default + Eq + Hash;
}

impl<G: bindings::GpuBindings> Gpu for G {
    type Error = G::Err;
    type DevicePtr = G::Ptr;
}

/// A GPU device, allowing the safe management of device streams
pub struct GpuDevice<G: Gpu> {
    ordinal: i32,
    context: G::Ctx,
    device: G::Dev,
}

unsafe impl<G: Gpu> Send for GpuDevice<G> {}
unsafe impl<G: Gpu> Sync for GpuDevice<G> {}

impl<G: Gpu> Drop for GpuDevice<G> {
    fn drop(&mut self) {
        unsafe {
            let _ = G::context_destroy(self.device);
        }
    }
}

impl<G: Gpu> GpuDevice<G> {
    pub fn new(ordinal: i32) -> Result<Arc<Self>, G::Error> {
        unsafe {
            G::driver_init()?;
        }

        let device = unsafe { G::device_get(ordinal)? };
        let context = unsafe { G::context_create(device)? };

        Ok(Arc::new(Self { ordinal, context, device }))
    }

    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    /// Set this device as currently active for this thread,
    /// which should be done before calling most runtime functions
    pub fn set(&self) -> Result<(), G::Error> {
        unsafe { G::context_set(self.context) }
    }

    /// Create a new stream on this device
    pub fn new_stream(self: Arc<Self>) -> Result<Arc<GpuStream<G>>, G::Error> {
        GpuStream::new(self.clone())
    }
}

/// A GPU stream associated with a GPU device
///
/// Safely handles the stream creation, destruction and syncing,
/// but exposes the raw unsafe stream operations such as allocating,
/// copying and launching kernels
pub struct GpuStream<G: Gpu> {
    id: usize,
    inner: G::Stream,
    device: Arc<GpuDevice<G>>,
}

impl<G: Gpu> Drop for GpuStream<G> {
    fn drop(&mut self) {
        self.sync().unwrap();
        unsafe { G::stream_destroy(self.inner).unwrap() };
    }
}

impl<G: Gpu> PartialEq for GpuStream<G> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<G: Gpu> Eq for GpuStream<G> {}

impl<G: Gpu> GpuStream<G> {
    /// The device that this stream resides on
    pub fn device(&self) -> Arc<GpuDevice<G>> {
        self.device.clone()
    }

    /// Created a new stream on the given `device`
    pub fn new(device: Arc<GpuDevice<G>>) -> Result<Arc<Self>, G::Error> {
        device.set()?;

        let inner = unsafe { G::stream_create()? };

        static ID: AtomicUsize = AtomicUsize::new(0);

        Ok(Arc::new(Self { id: ID.fetch_add(1, Ordering::SeqCst), inner, device }))
    }

    /// Block the host thread until all queued operations on this
    /// stream have completed
    pub fn sync(&self) -> Result<(), G::Error> {
        self.device.set()?;
        unsafe { G::stream_sync(self.inner) }
    }

    /// Queue allocating `bytes` amount of memory on this stream
    pub fn malloc(&self, bytes: usize) -> Result<G::DevicePtr, G::Error> {
        self.device.set()?;
        unsafe { G::stream_malloc(self.inner, bytes) }
    }

    /// Queue freeing the given device pointer on this stream
    ///
    /// ### Safety
    ///
    /// User must ensure `ptr` is pointing to a valid device allocation
    pub unsafe fn free(&self, ptr: G::DevicePtr) -> Result<(), G::Error> {
        self.device.set()?;
        unsafe { G::stream_free(self.inner, ptr) }
    }

    /// Queue setting each byte in the `bytes` amount of memory on the
    /// device starting at `ptr` to `value` on this stream
    ///
    /// ### Safety
    ///
    /// User must ensure that `ptr` is pointing to a valid allocation
    /// on the device
    ///
    /// User must ensure that `ptr` remains valid until the memset has
    /// been ensured to have completed via a stream sync
    pub unsafe fn memset(&self, ptr: G::DevicePtr, bytes: usize, value: u8) -> Result<(), G::Error> {
        self.device.set()?;
        unsafe { G::stream_memset(self.inner, ptr, bytes, value) }
    }

    /// Queue a copy of `bytes` amount of memory from `src` to `dst`
    /// on this stream, where `src` and `dst` resides on host/device
    /// respectively
    ///
    /// ### Safety
    ///
    /// User must ensure that `src` and `dst` are pointing to valid
    /// allocations on the respective host/device
    ///
    /// User must ensure that `src` and `dst` remain valid until the
    /// copy has been ensured to have completed via a stream sync
    pub unsafe fn memcpy_h2d(&self, src: *const c_void, dst: G::DevicePtr, bytes: usize) -> Result<(), G::Error> {
        self.device.set()?;
        unsafe { G::stream_memcpy_h2d(self.inner, dst, src, bytes) }
    }

    /// Queue a copy of `bytes` amount of memory from `src` to `dst`
    /// on this stream, where `src` and `dst` resides on device/host
    /// respectively
    ///
    /// ### Safety
    ///
    /// User must ensure that `src` and `dst` are pointing to valid
    /// allocations on the respective device/host
    ///
    /// User must ensure that `src` and `dst` remain valid until the
    /// copy has been ensured to have completed via a stream sync
    pub unsafe fn memcpy_d2h(&self, src: G::DevicePtr, dst: *mut c_void, bytes: usize) -> Result<(), G::Error> {
        self.device.set()?;
        unsafe { G::stream_memcpy_d2h(self.inner, dst, src, bytes) }
    }
}

pub struct GpuModule<G: Gpu> {
    module: G::Module,
    device: Arc<GpuDevice<G>>,
}

impl<G: Gpu> Drop for GpuModule<G> {
    fn drop(&mut self) {
        unsafe { G::module_destroy(self.module).unwrap() }
    }
}

impl<G: Gpu> GpuModule<G> {
    /// Compiles the source code and loads the resulting module
    /// onto the device
    pub fn new(device: Arc<GpuDevice<G>>, source_code: impl Into<String>) -> Result<Arc<Self>, G::Error> {
        let src = CString::new(source_code.into()).map_err(|e| format!("{e:?}"))?;

        device.set()?;

        let code = unsafe { G::program_compile(&src, 0, std::ptr::null())? };
        let module = unsafe { G::module_create(code.as_ptr().cast())? };

        Ok(Arc::new(Self { device, module }))
    }

    /// Get kernel with given name from module
    pub fn get_kernel(self: Arc<Self>, name: impl Into<String>) -> Result<GpuKernel<G>, G::Error> {
        let name = CString::new(name.into()).map_err(|e| format!("{e:?}"))?;
        let kernel = unsafe { G::module_get_kernel(self.module, &name)? };

        unsafe {
            G::kernel_load(kernel)?;
        }

        Ok(GpuKernel { kernel, module: self.clone() })
    }

    /// Get device that this module is on
    pub fn device(&self) -> Arc<GpuDevice<G>> {
        self.device.clone()
    }
}

pub struct GpuKernel<G: Gpu> {
    kernel: G::Kernel,
    module: Arc<GpuModule<G>>,
}

impl<G: Gpu> GpuKernel<G> {
    /// Queue a GPU kernel for execution on this stream.
    ///
    /// ### Safety
    ///
    /// User must ensure that invoking the provided kernel with
    /// the given arguments does not invoke undefined behaviour
    ///
    /// User must ensure that all device pointers in the `args`
    /// remain valid until the kernel has been ensured to have
    /// completed via a stream sync
    pub unsafe fn launch(
        &self,
        stream: &GpuStream<G>,
        grid_dim: Dim3,
        block_size: u32,
        args: *mut *mut c_void,
        smem: u32,
    ) -> Result<(), G::Error> {
        let o1 = stream.device.ordinal();
        let o2 = self.module.device.ordinal();
        if o1 != o2 {
            return Err(format!("Attempted to launch GPU{o1} kernel on GPU{o2}!").into());
        }

        if block_size > 1024 {
            return Err(format!("Attempted to launch kernel with {block_size} > 1024 threads per block!").into());
        }

        let block_dim = Dim3 { x: block_size, y: 1, z: 1 };

        stream.device.set()?;
        unsafe { G::kernel_launch(self.kernel, stream.inner, grid_dim, block_dim, args, smem) }
    }

    /// Get the device that this kernel is on
    pub fn device(&self) -> Arc<GpuDevice<G>> {
        self.module.device()
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
#[cfg(test)]
mod tests {
    use super::*;

    fn create_malloc_copy_sync_drop<G: Gpu>() -> Result<(), G::Error> {
        let host_src = [1.0f32, 2.0, 3.0, 4.0];
        let mut host_dst = [0.0, 0.0, 0.0, 0.0];

        let device = GpuDevice::<G>::new(0)?;
        let stream = GpuStream::new(device.clone())?;

        unsafe {
            let dev_ptr = stream.malloc(16)?;
            stream.sync()?;
            stream.memcpy_h2d(host_src.as_ptr().cast(), dev_ptr, 16)?;
            stream.memcpy_d2h(dev_ptr, host_dst.as_mut_ptr().cast(), 16)?;
            stream.sync()?;
        }

        assert_eq!(host_dst, host_src);

        Ok(())
    }

    fn multiple_device_instances<G: Gpu>() -> Result<(), G::Error> {
        let a = GpuDevice::<G>::new(0)?;
        let sa = GpuStream::new(a.clone())?;
        let b = GpuDevice::<G>::new(0)?;
        let sb = GpuStream::new(b.clone())?;

        drop(sa);
        drop(a);
        drop(sb);
        drop(b);

        Ok(())
    }

    fn compile_load_execute_kernel<G: Gpu>() -> Result<(), G::Error> {
        let host_src = [1.0f32, 2.0, 3.0, 4.0];
        let mut host_dst = [0.0, 0.0, 0.0, 0.0];

        let device = GpuDevice::<G>::new(0)?;
        let stream = GpuStream::new(device.clone())?;

        let add_one_kernel = "
            extern \"C\" __global__ void kernel(const int size, const float* src, float* dst) {
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid < size) dst[tid] = src[tid] + 1.0;
            }
        ";

        let module = GpuModule::new(device.clone(), add_one_kernel)?;
        let kernel = module.get_kernel("kernel")?;

        unsafe {
            let dev_src = stream.malloc(16)?;
            stream.sync()?;

            let dev_dst = stream.malloc(16)?;
            stream.sync()?;

            stream.memcpy_h2d(host_src.as_ptr().cast(), dev_src, 16)?;

            let gdim = Dim3 { x: 1, y: 1, z: 1 };
            let size = 4i32;
            let mut args = Vec::new();
            args.push((&size) as *const i32 as *mut c_void);
            args.push((&dev_src) as *const G::DevicePtr as *mut c_void);
            args.push((&dev_dst) as *const G::DevicePtr as *mut c_void);

            kernel.launch(&stream, gdim, 4, args.as_mut_ptr(), 0)?;

            stream.memcpy_d2h(dev_dst, host_dst.as_mut_ptr().cast(), 16)?;
            stream.sync()?;
        }

        for (d, s) in host_dst.into_iter().zip(host_src) {
            assert_eq!(d, s + 1.0);
        }

        Ok(())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use crate::runtime::cuda::{Cuda, CudaError};

        #[test]
        fn create_malloc_copy_sync_drop() -> Result<(), CudaError> {
            super::create_malloc_copy_sync_drop::<Cuda>()
        }

        #[test]
        fn multiple_device_instances() -> Result<(), CudaError> {
            super::multiple_device_instances::<Cuda>()
        }

        #[test]
        fn compile_load_execute_kernel() -> Result<(), CudaError> {
            super::compile_load_execute_kernel::<Cuda>()
        }
    }

    #[cfg(feature = "rocm")]
    mod rocm {
        use crate::runtime::rocm::{ROCm, ROCmError};

        #[test]
        fn create_malloc_copy_sync_drop() -> Result<(), ROCmError> {
            super::create_malloc_copy_sync_drop::<ROCm>()
        }

        #[test]
        fn multiple_device_instances() -> Result<(), ROCmError> {
            super::multiple_device_instances::<ROCm>()
        }

        #[test]
        fn compile_load_execute_kernel() -> Result<(), ROCmError> {
            super::compile_load_execute_kernel::<ROCm>()
        }
    }
}
