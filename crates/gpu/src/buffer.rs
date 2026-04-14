//! Simple GPU buffer, managed by a guard and a stream-syncing wrapper

use std::{
    ops::Deref,
    sync::{Arc, Mutex},
};

use bullet_compiler::tensor::{DType, TValue};

use crate::runtime::{Device, Gpu, Stream};

/// Keeps buffer ownership alive on a given stream until
/// dropped, at which point it syncs the stream
pub struct SyncOnDrop<G: Gpu> {
    stream: Arc<Stream<G>>,
    guards: Vec<BufferGuard<G>>,
    synced: bool,
}

impl<G: Gpu> Drop for SyncOnDrop<G> {
    fn drop(&mut self) {
        if !self.synced {
            self.stream.sync().unwrap();
        }
    }
}

impl<G: Gpu> SyncOnDrop<G> {
    pub fn new(stream: Arc<Stream<G>>) -> Self {
        Self { stream, guards: Vec::new(), synced: false }
    }

    pub fn sync(self) -> Result<(), G::Error> {
        self.stream.sync()
    }

    pub fn stream(&self) -> Arc<Stream<G>> {
        self.stream.clone()
    }

    pub fn guards(&self) -> &[BufferGuard<G>] {
        &self.guards
    }

    pub fn attach(&mut self, guard: BufferGuard<G>) -> Result<(), G::Error> {
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
    pub fn value(self) -> Result<T, G::Error> {
        self.sync.sync()?;
        Ok(self.value)
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

    /// Drop the stored value, returning the SyncOnDrop only (so
    /// not syncing)
    ///
    /// ### Safety
    ///
    /// This is marked as blanket "unsafe" because some values may
    /// require stream syncing before they can legally be dropped
    pub unsafe fn detach_value(self) -> SyncOnDrop<G> {
        self.sync
    }

    pub fn stream(&self) -> Arc<Stream<G>> {
        self.sync.stream()
    }

    pub fn guards(&self) -> &[BufferGuard<G>] {
        self.sync.guards()
    }

    pub fn attach_guard(&mut self, guard: BufferGuard<G>) {
        self.sync.guards.push(guard);
    }

    pub fn attach_value<U>(self, other: U) -> SyncOnValue<G, (T, U)> {
        SyncOnValue { sync: self.sync, value: (self.value, other) }
    }
}

pub struct Buffer<G: Gpu> {
    ptr: G::DevicePtr,
    dtype: DType,
    size: usize,
    device: Arc<Device<G>>,
    creator: Option<Arc<Stream<G>>>,
    owner: Mutex<Option<(Arc<Stream<G>>, usize)>>,
}

impl<G: Gpu> Drop for Buffer<G> {
    fn drop(&mut self) {
        unsafe {
            if let Some(stream) = &self.creator {
                stream.free(self.ptr).unwrap();
            } else {
                self.device.free(self.ptr).unwrap();
            }
        }
    }
}

impl<G: Gpu> Buffer<G> {
    /// New uninitialised buffer on the device with given size and dtype
    ///
    /// ### Safety
    ///
    /// The user must ensure that the memory is initialised before it
    /// is ever read
    pub unsafe fn uninit(device: &Arc<Device<G>>, dtype: DType, size: usize) -> Result<Arc<Self>, G::Error> {
        if size == 0 {
            return Err("Attempted to allocated 0-size device memory!".to_string().into());
        }

        let ptr = device.malloc(dtype.bytes() * size)?;

        Ok(Arc::new(Self { ptr, dtype, size, device: device.clone(), creator: None, owner: Mutex::new(None) }))
    }

    /// New zeroed buffer on the device with given size and dtype
    pub fn zeroed(device: &Arc<Device<G>>, dtype: DType, size: usize) -> Result<Arc<Self>, G::Error> {
        unsafe {
            let buf = Self::uninit(device, dtype, size)?;
            device.memset(buf.ptr, buf.bytes(), 0)?;
            Ok(buf)
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

    pub fn device(&self) -> Arc<Device<G>> {
        self.device.clone()
    }

    pub fn owner(&self) -> Option<Arc<Stream<G>>> {
        self.owner.lock().unwrap().as_ref().map(|x| x.0.clone())
    }

    /// Take "ownership" of this buffer with the given stream,
    /// allowing acces to the raw device pointer
    ///
    /// Returns an error if this buffer is already owned by a
    /// different stream
    pub fn acquire(self: &Arc<Self>, stream: Arc<Stream<G>>) -> Result<BufferGuard<G>, G::Error> {
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

        Ok(BufferGuard(self.clone()))
    }

    /// Copy buffer to host on the given stream
    pub fn to_host_async(self: &Arc<Self>, stream: &Arc<Stream<G>>) -> Result<SyncOnValue<G, TValue>, G::Error> {
        let guard = self.clone().acquire(stream.clone())?;
        let mut value = TValue::zeros(guard.dtype, guard.size);

        unsafe {
            stream.memcpy_d2h(guard.ptr(), value.mut_ptr(), guard.bytes())?;
        }

        let mut sync = SyncOnDrop::new(stream.clone());
        sync.attach(guard)?;

        Ok(SyncOnValue::new(sync, value))
    }

    /// Copy buffer to host
    pub fn to_host(self: &Arc<Self>) -> Result<TValue, G::Error> {
        let mut value = TValue::zeros(self.dtype, self.size);

        unsafe {
            self.device.memcpy_d2h(self.ptr, value.mut_ptr(), self.bytes())?;
        }

        Ok(value)
    }

    /// Create buffer from host values on the given device
    pub fn from_host(device: &Arc<Device<G>>, value: &TValue) -> Result<Arc<Self>, G::Error> {
        unsafe {
            let buf = Self::uninit(device, value.dtype(), value.size())?;
            device.memcpy_h2d(value.ptr(), buf.ptr, buf.bytes())?;
            Ok(buf)
        }
    }

    pub fn copy_from_host_async<'a>(
        self: &Arc<Self>,
        stream: &Arc<Stream<G>>,
        value: &'a TValue,
    ) -> Result<SyncOnValue<G, &'a TValue>, G::Error> {
        if self.size() != value.size() {
            return Err(format!("Mismatched sizes: {} != {}", self.size(), value.size()).into());
        }

        if self.dtype() != value.dtype() {
            return Err(format!("Mismatched DType: {:?} != {:?}", self.dtype(), value.dtype()).into());
        }

        unsafe {
            let mut sync = SyncOnDrop::new(stream.clone());
            let guard = self.clone().acquire(stream.clone())?;

            stream.memcpy_h2d(value.ptr(), guard.ptr(), guard.bytes())?;

            sync.attach(guard)?;
            Ok(SyncOnValue::new(sync, value))
        }
    }

    pub fn copy_from_host(self: &Arc<Self>, value: &TValue) -> Result<(), G::Error> {
        if self.size() != value.size() {
            return Err(format!("Mismatched sizes: {} != {}", self.size(), value.size()).into());
        }

        if self.dtype() != value.dtype() {
            return Err(format!("Mismatched DType: {:?} != {:?}", self.dtype(), value.dtype()).into());
        }

        unsafe { self.device.memcpy_h2d(value.ptr(), self.ptr, self.bytes()) }
    }
}

pub struct BufferGuard<G: Gpu>(Arc<Buffer<G>>);

impl<G: Gpu> Deref for BufferGuard<G> {
    type Target = Arc<Buffer<G>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<G: Gpu> Drop for BufferGuard<G> {
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

impl<G: Gpu> BufferGuard<G> {
    pub fn ptr(&self) -> G::DevicePtr {
        self.0.ptr
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
#[cfg(test)]
mod tests {
    use crate::runtime::Device;

    use super::*;

    fn from_to_host<G: Gpu>() -> Result<(), G::Error> {
        let host_src = TValue::F32(vec![1.0, 2.0, 3.0, 4.0]);

        let device = Device::<G>::new(0)?;

        let buf = Buffer::from_host(&device, &host_src)?;
        let host_dst = buf.to_host()?;

        assert_eq!(host_src, host_dst);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use crate::runtime::cuda::{Cuda, CudaError};

        #[test]
        fn from_to_host() -> Result<(), CudaError> {
            super::from_to_host::<Cuda>()
        }
    }

    #[cfg(feature = "rocm")]
    mod rocm {
        use crate::runtime::rocm::{ROCm, ROCmError};

        #[test]
        fn from_to_host() -> Result<(), ROCmError> {
            super::from_to_host::<ROCm>()
        }
    }
}
