use std::{ffi::c_void, fmt, marker::PhantomData, mem::MaybeUninit, sync::Arc};

use crate::gpu::bindings;
pub use crate::gpu::bindings::{Dim3, MemcpyKind};

#[cfg(feature = "cuda")]
pub use crate::gpu::bindings::cuda;

#[cfg(feature = "hip")]
pub use crate::gpu::bindings::hip;

pub trait Gpu: bindings::GpuBindings<E = Self::Error, S = Self::Stream> {
    type Error: Copy + fmt::Debug + Eq;
    type Stream: Copy + fmt::Debug + Eq;
}

impl<T: bindings::GpuBindings> Gpu for T {
    type Error = T::E;
    type Stream = T::S;
}

pub struct GpuDevice<T: Gpu>(i32, PhantomData<T>);

unsafe impl<T: Gpu> Send for GpuDevice<T> {}
unsafe impl<T: Gpu> Sync for GpuDevice<T> {}

impl<T: Gpu> GpuDevice<T> {
    pub fn new(ordinal: i32) -> Result<Arc<Self>, T::Error> {
        unsafe {
            T::device_init(ordinal)?;
        }

        Ok(Arc::new(Self(ordinal, PhantomData)))
    }

    pub fn ordinal(&self) -> i32 {
        self.0
    }

    /// Set this device as currently active for this thread,
    /// which should be done before calling most runtime functions
    pub fn set(&self) -> Result<(), T::Error> {
        unsafe { T::device_set(self.0) }
    }
}

pub struct GpuStream<T: Gpu> {
    inner: T::Stream,
    device: Arc<GpuDevice<T>>,
}

impl<T: Gpu> Drop for GpuStream<T> {
    fn drop(&mut self) {
        self.sync().unwrap();
        unsafe { T::stream_destroy(self.inner).unwrap() };
    }
}

impl<T: Gpu> GpuStream<T> {
    /// The device that this stream is associated with
    pub fn device(&self) -> Arc<GpuDevice<T>> {
        self.device.clone()
    }

    /// Created a new stream associated with the given `device`
    pub fn new(device: Arc<GpuDevice<T>>) -> Result<Arc<Self>, T::Error> {
        device.set()?;

        let inner = unsafe {
            let mut uninit = MaybeUninit::uninit();
            T::stream_create(uninit.as_mut_ptr())?;
            uninit.assume_init()
        };

        Ok(Arc::new(Self { inner, device }))
    }

    /// Block the host thread until all queued operations on this
    /// stream have completed
    pub fn sync(&self) -> Result<(), T::Error> {
        self.device.set()?;
        unsafe { T::stream_sync(self.inner) }
    }

    /// Queue allocating `bytes` amount of memory on this stream,
    /// storing the resulting pointer in `ptr`
    ///
    /// ### Safety
    /// User must ensure `ptr` is not dereferenced before the malloc
    /// is ensured to have been completed via a stream sync
    ///
    /// User can pass `ptr` to functions executed on this stream as
    /// sequential execution is guaranteed
    pub unsafe fn malloc(&self, ptr: *mut *mut c_void, bytes: usize) -> Result<(), T::Error> {
        self.device.set()?;
        unsafe { T::stream_malloc(self.inner, ptr, bytes) }
    }

    /// Queue freeing the given device pointer on this stream
    ///
    /// ### Safety
    /// User must ensure `ptr` is pointing to a valid device allocation
    pub unsafe fn free(&self, ptr: *mut c_void) -> Result<(), T::Error> {
        self.device.set()?;
        unsafe { T::stream_free(self.inner, ptr) }
    }

    /// Queue a copy of `bytes` amount of memory from `src` to `dst`
    /// on this stream, where `src` and `dst` resides on host/device
    /// as specified by `kind`
    ///
    /// ### Safety
    /// User must ensure that `src` and `dst` must be pointing to
    /// valid allocations on the respective host/device as specified
    /// by `kind`
    ///
    /// User must ensure that `src` and `dst` remain valid until the
    /// copy has been ensured to have completed via a stream sync
    pub unsafe fn memcpy(
        &self,
        src: *const c_void,
        dst: *mut c_void,
        bytes: usize,
        kind: MemcpyKind,
    ) -> Result<(), T::Error> {
        self.device.set()?;
        unsafe { T::stream_memcpy(self.inner, dst, src, bytes, kind) }
    }

    /// Queue a GPU kernel for execution on this stream.
    ///
    /// ### Safety
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
    ) -> Result<(), T::Error> {
        self.device.set()?;
        unsafe { T::stream_launch_kernel(self.inner, func, drid_dim, block_dim, args, smem) }
    }
}

#[cfg(any(feature = "cuda", feature = "hip"))]
#[cfg(test)]
mod tests {
    use super::*;

    fn basics<T: Gpu>() -> Result<(), T::Error> {
        use std::mem::MaybeUninit;

        let host_src = [1.0f32, 2.0, 3.0, 4.0];
        let mut host_dst = [0.0, 0.0, 0.0, 0.0];

        let device = GpuDevice::<T>::new(0)?;
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

    fn multiple_device_instances<T: Gpu>() -> Result<(), T::Error> {
        let a = GpuDevice::<T>::new(0)?;
        let sa = GpuStream::new(a.clone())?;
        let b = GpuDevice::<T>::new(0)?;
        let sb = GpuStream::new(b.clone())?;

        drop(sa);
        drop(a);
        drop(sb);
        drop(b);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use crate::gpu::device::cuda::{Cuda, CudaError};

        #[test]
        fn basics() -> Result<(), CudaError> {
            super::basics::<Cuda>()
        }

        #[test]
        fn multiple_device_instances() -> Result<(), CudaError> {
            super::multiple_device_instances::<Cuda>()
        }
    }

    #[cfg(feature = "hip")]
    mod hip {
        use crate::gpu::device::hip::{Hip, HipError};

        #[test]
        fn basics() -> Result<(), HipError> {
            super::basics::<Hip>()
        }

        #[test]
        fn multiple_device_instances() -> Result<(), HipError> {
            super::multiple_device_instances::<Hip>()
        }
    }
}
