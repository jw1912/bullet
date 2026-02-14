//! Minimal wrapper around CUDA/ROCm devices and streams

mod bindings;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "rocm")]
pub mod rocm;

use std::{
    ffi::c_void,
    fmt,
    marker::PhantomData,
    mem::MaybeUninit,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

pub use bindings::{Dim3, MemcpyKind};

pub trait Gpu: bindings::GpuBindings<E = Self::Error, S = Self::Stream> {
    type Error: fmt::Debug + Eq + From<String>;
    type Stream: Copy + fmt::Debug + Eq;
}

impl<G: bindings::GpuBindings> Gpu for G {
    type Error = G::E;
    type Stream = G::S;
}

pub struct GpuDevice<G: Gpu>(i32, PhantomData<G>);

unsafe impl<G: Gpu> Send for GpuDevice<G> {}
unsafe impl<G: Gpu> Sync for GpuDevice<G> {}

impl<G: Gpu> GpuDevice<G> {
    pub fn new(ordinal: i32) -> Result<Arc<Self>, G::Error> {
        unsafe {
            G::device_init(ordinal)?;
        }

        Ok(Arc::new(Self(ordinal, PhantomData)))
    }

    pub fn ordinal(&self) -> i32 {
        self.0
    }

    /// Set this device as currently active for this thread,
    /// which should be done before calling most runtime functions
    pub fn set(&self) -> Result<(), G::Error> {
        unsafe { G::device_set(self.0) }
    }
}

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
    /// The device that this stream is associated with
    pub fn device(&self) -> Arc<GpuDevice<G>> {
        self.device.clone()
    }

    /// Created a new stream associated with the given `device`
    pub fn new(device: Arc<GpuDevice<G>>) -> Result<Arc<Self>, G::Error> {
        device.set()?;

        let inner = unsafe {
            let mut uninit = MaybeUninit::uninit();
            G::stream_create(uninit.as_mut_ptr())?;
            uninit.assume_init()
        };

        static ID: AtomicUsize = AtomicUsize::new(0);

        Ok(Arc::new(Self { id: ID.fetch_add(1, Ordering::SeqCst), inner, device }))
    }

    /// Block the host thread until all queued operations on this
    /// stream have completed
    pub fn sync(&self) -> Result<(), G::Error> {
        self.device.set()?;
        unsafe { G::stream_sync(self.inner) }
    }

    /// Queue allocating `bytes` amount of memory on this stream,
    /// storing the resulting pointer in `ptr`
    ///
    /// ### Safety
    ///
    /// User must ensure `ptr` is not dereferenced before the malloc
    /// is ensured to have been completed via a stream sync
    ///
    /// User can pass `ptr` to functions executed on this stream as
    /// sequential execution is guaranteed
    pub unsafe fn malloc(&self, ptr: *mut *mut c_void, bytes: usize) -> Result<(), G::Error> {
        self.device.set()?;
        unsafe { G::stream_malloc(self.inner, ptr, bytes) }
    }

    /// Queue freeing the given device pointer on this stream
    ///
    /// ### Safety
    ///
    /// User must ensure `ptr` is pointing to a valid device allocation
    pub unsafe fn free(&self, ptr: *mut c_void) -> Result<(), G::Error> {
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
    pub unsafe fn memset(&self, ptr: *mut c_void, bytes: usize, value: u8) -> Result<(), G::Error> {
        self.device.set()?;
        unsafe { G::stream_memset(self.inner, ptr, bytes, value) }
    }

    /// Queue a copy of `bytes` amount of memory from `src` to `dst`
    /// on this stream, where `src` and `dst` resides on host/device
    /// as specified by `kind`
    ///
    /// ### Safety
    ///
    /// User must ensure that `src` and `dst` are pointing to valid
    /// allocations on the respective host/device as specified by `kind`
    ///
    /// User must ensure that `src` and `dst` remain valid until the
    /// copy has been ensured to have completed via a stream sync
    pub unsafe fn memcpy(
        &self,
        src: *const c_void,
        dst: *mut c_void,
        bytes: usize,
        kind: MemcpyKind,
    ) -> Result<(), G::Error> {
        self.device.set()?;
        unsafe { G::stream_memcpy(self.inner, dst, src, bytes, kind) }
    }

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
    pub unsafe fn launch_kernel(
        &self,
        func: *const c_void,
        drid_dim: Dim3,
        block_dim: Dim3,
        args: *mut *mut c_void,
        smem: usize,
    ) -> Result<(), G::Error> {
        self.device.set()?;
        unsafe { G::stream_launch_kernel(self.inner, func, drid_dim, block_dim, args, smem) }
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
#[cfg(test)]
mod tests {
    use super::*;

    fn create_malloc_copy_sync_drop<G: Gpu>() -> Result<(), G::Error> {
        use std::mem::MaybeUninit;

        let host_src = [1.0f32, 2.0, 3.0, 4.0];
        let mut host_dst = [0.0, 0.0, 0.0, 0.0];

        let device = GpuDevice::<G>::new(0)?;
        let stream = GpuStream::new(device.clone())?;

        unsafe {
            let mut dev_ptr = MaybeUninit::uninit();
            stream.malloc(dev_ptr.as_mut_ptr(), 16)?;
            stream.sync()?;
            let dev_ptr = dev_ptr.assume_init();
            stream.memcpy(host_src.as_ptr().cast(), dev_ptr, 16, MemcpyKind::HostToDevice)?;
            stream.memcpy(dev_ptr, host_dst.as_mut_ptr().cast(), 16, MemcpyKind::DeviceToHost)?;
            stream.sync()?;
        }

        assert_eq!(host_dst, host_src);

        drop(stream);
        drop(device);

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

    #[cfg(feature = "cuda")]
    mod cuda {
        use crate::device::cuda::{Cuda, CudaError};

        #[test]
        fn create_malloc_copy_sync_drop() -> Result<(), CudaError> {
            super::create_malloc_copy_sync_drop::<Cuda>()
        }

        #[test]
        fn multiple_device_instances() -> Result<(), CudaError> {
            super::multiple_device_instances::<Cuda>()
        }
    }

    #[cfg(feature = "rocm")]
    mod rocm {
        use crate::device::rocm::{ROCm, ROCmError};

        #[test]
        fn create_malloc_copy_sync_drop() -> Result<(), ROCmError> {
            super::create_malloc_copy_sync_drop::<ROCm>()
        }

        #[test]
        fn multiple_device_instances() -> Result<(), ROCmError> {
            super::multiple_device_instances::<ROCm>()
        }
    }
}
