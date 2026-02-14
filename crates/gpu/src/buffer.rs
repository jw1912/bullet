use std::{
    ffi::c_void,
    mem::MaybeUninit,
    ops::Deref,
    sync::{Arc, Mutex},
};

use bullet_compiler::graph::{DType, TValue};

use crate::device::{Gpu, GpuStream, MemcpyKind};

/// Keeps buffer ownership alive on a given stream until
/// dropped, at which point it syncs the stream
pub struct SyncOnDrop<G: Gpu> {
    stream: Arc<GpuStream<G>>,
    guards: Vec<GpuBufferGuard<G>>,
}

impl<G: Gpu> Drop for SyncOnDrop<G> {
    fn drop(&mut self) {
        self.stream.sync().unwrap();
    }
}

impl<G: Gpu> SyncOnDrop<G> {
    pub fn new(stream: Arc<GpuStream<G>>) -> Self {
        Self { stream, guards: Vec::new() }
    }

    pub fn stream(&self) -> Arc<GpuStream<G>> {
        self.stream.clone()
    }

    pub fn guards(&self) -> &[GpuBufferGuard<G>] {
        &self.guards
    }

    pub fn attach(&mut self, guard: GpuBufferGuard<G>) -> Result<(), G::Error> {
        if guard.owner().unwrap() != self.stream {
            return Err("Guard is owned by a different stream!".to_string().into());
        }

        self.guards.push(guard);

        Ok(())
    }
}

/// Like `SyncOnDrop` but with an attached value
pub struct SyncOnValue<G: Gpu, T> {
    sync: SyncOnDrop<G>,
    value: T,
}

impl<G: Gpu, T> SyncOnValue<G, T> {
    pub fn new(sync: SyncOnDrop<G>, value: T) -> Self {
        Self { sync, value }
    }

    /// Consume self, which causes a stream sync, and return
    /// stored value
    pub fn value(self) -> T {
        self.value
    }

    /// Get a reference to the stored value
    ///
    /// ### Safety
    ///
    /// This is marked as blanket "unsafe" because some values may
    /// require stream syncing before they can legally be accessed
    pub unsafe fn value_ref(&self) -> &T {
        &self.value
    }

    pub fn stream(&self) -> Arc<GpuStream<G>> {
        self.sync.stream()
    }

    pub fn guards(&self) -> &[GpuBufferGuard<G>] {
        self.sync.guards()
    }

    pub fn attach_guard(&mut self, guard: GpuBufferGuard<G>) {
        self.sync.guards.push(guard);
    }

    pub fn attach_value<U>(self, other: U) -> SyncOnValue<G, (T, U)> {
        SyncOnValue { sync: self.sync, value: (self.value, other) }
    }
}

pub struct GpuBuffer<G: Gpu> {
    ptr: *mut c_void,
    dtype: DType,
    size: usize,
    creator: Arc<GpuStream<G>>,
    owner: Mutex<Option<(Arc<GpuStream<G>>, usize)>>,
}

impl<G: Gpu> Drop for GpuBuffer<G> {
    fn drop(&mut self) {
        unsafe {
            self.creator.free(self.ptr).unwrap();
        }
    }
}

impl<G: Gpu> GpuBuffer<G> {
    /// New uninitialised buffer on the device, allocated by `stream`,
    /// with given size and dtype
    ///
    /// ### Safety
    ///
    /// The user must ensure that the memory is initialised before it
    /// is ever read
    pub unsafe fn uninit(
        stream: Arc<GpuStream<G>>,
        dtype: DType,
        size: usize,
    ) -> Result<SyncOnValue<G, Arc<Self>>, G::Error> {
        if size == 0 {
            return Err("Attempted to allocated 0-size device memory!".to_string().into());
        }

        let ptr = unsafe {
            let mut ptr = MaybeUninit::uninit();
            stream.malloc(ptr.as_mut_ptr(), dtype.bytes() * size)?;
            ptr.assume_init()
        };

        if ptr.is_null() {
            return Err("Allocated pointer was null!".to_string().into());
        }

        if ptr.align_offset(dtype.bytes()) != 0 {
            return Err("Device allocated pointer is not appropriately aligned!".to_string().into());
        }

        let buf = Arc::new(Self { ptr, dtype, size, creator: stream.clone(), owner: Mutex::new(None) });
        let mut sync = SyncOnDrop::new(stream.clone());
        sync.attach(buf.clone().acquire(stream)?)?;

        Ok(SyncOnValue::new(sync, buf))
    }

    /// New zeroed buffer on the device with given size and dtype
    pub fn zeroed(stream: Arc<GpuStream<G>>, dtype: DType, size: usize) -> Result<SyncOnValue<G, Arc<Self>>, G::Error> {
        unsafe {
            let sync = Self::uninit(stream.clone(), dtype, size)?;
            let aqcuired = sync.value_ref().clone().acquire(stream.clone())?;
            stream.memset(aqcuired.ptr(), aqcuired.bytes(), 0)?;
            Ok(sync)
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Size of the buffer in raw bytes
    pub fn bytes(&self) -> usize {
        self.dtype.bytes() * self.size
    }

    pub fn owner(&self) -> Option<Arc<GpuStream<G>>> {
        self.owner.lock().unwrap().as_ref().map(|x| x.0.clone())
    }

    /// Take "ownership" of this buffer with the given stream,
    /// allowing acces to the raw device pointer
    ///
    /// Returns an error if this buffer is already owned by a
    /// different stream
    pub fn acquire(self: Arc<Self>, stream: Arc<GpuStream<G>>) -> Result<GpuBufferGuard<G>, G::Error> {
        let mut owner = self.owner.lock().unwrap();

        if let Some((owning, count)) = owner.as_mut() {
            if owning.as_ref() == stream.as_ref() {
                *count += 1;
            } else {
                return Err("Buffer is already owned!".to_string().into());
            }
        } else {
            *owner = Some((stream, 1));
        }

        drop(owner);

        Ok(GpuBufferGuard(self))
    }

    pub fn to_host(self: Arc<Self>, stream: Arc<GpuStream<G>>) -> Result<SyncOnValue<G, TValue>, G::Error> {
        let guard = self.acquire(stream.clone())?;
        let mut value = TValue::zeros(guard.dtype, guard.size);

        unsafe {
            stream.memcpy(guard.ptr(), value.mut_ptr(), guard.bytes(), MemcpyKind::DeviceToHost)?;
        }

        let mut sync = SyncOnDrop::new(stream);
        sync.attach(guard)?;

        Ok(SyncOnValue::new(sync, value))
    }

    #[allow(clippy::type_complexity)]
    pub fn from_host(
        stream: Arc<GpuStream<G>>,
        value: &TValue,
    ) -> Result<SyncOnValue<G, (Arc<Self>, &TValue)>, G::Error> {
        unsafe {
            let buf = Self::uninit(stream.clone(), value.dtype(), value.size())?;
            let guard = &buf.guards()[0];

            stream.memcpy(value.ptr(), guard.ptr(), guard.bytes(), MemcpyKind::HostToDevice)?;

            Ok(buf.attach_value(value))
        }
    }
}

pub struct GpuBufferGuard<G: Gpu>(Arc<GpuBuffer<G>>);

impl<G: Gpu> Deref for GpuBufferGuard<G> {
    type Target = Arc<GpuBuffer<G>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<G: Gpu> Drop for GpuBufferGuard<G> {
    fn drop(&mut self) {
        let mut owner = self.0.owner.lock().unwrap();

        if let Some((_, count)) = owner.as_mut() {
            *count -= 1;

            if *count == 0 {
                *owner = None;
            }
        }
    }
}

impl<G: Gpu> GpuBufferGuard<G> {
    pub fn ptr(&self) -> *mut c_void {
        self.0.ptr
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
#[cfg(test)]
mod tests {
    use crate::device::GpuDevice;

    use super::*;

    fn from_to_host<G: Gpu>() -> Result<(), G::Error> {
        let host_src = TValue::F32(vec![1.0, 2.0, 3.0, 4.0]);

        let device = GpuDevice::<G>::new(0)?;
        let stream = GpuStream::new(device.clone())?;

        let buf = GpuBuffer::from_host(stream.clone(), &host_src)?.value().0;
        let host_dst = buf.to_host(stream)?.value();

        assert_eq!(host_src, host_dst);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use crate::device::cuda::{Cuda, CudaError};

        #[test]
        fn from_to_host() -> Result<(), CudaError> {
            super::from_to_host::<Cuda>()
        }
    }

    #[cfg(feature = "rocm")]
    mod rocm {
        use crate::device::rocm::{ROCm, ROCmError};

        #[test]
        fn from_to_host() -> Result<(), ROCmError> {
            super::from_to_host::<ROCm>()
        }
    }
}
