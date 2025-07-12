use std::sync::Arc;

use bullet_core::device::DeviceBuffer;

use crate::DeviceError;

use super::ExecutionContext;

/// # Safety
/// Don't impl this for anything else
pub unsafe trait ValidType {}
unsafe impl ValidType for f32 {}
unsafe impl ValidType for i32 {}

/// Managed memory buffer of `T` on the device.
#[derive(Debug)]
pub struct Buffer<T: ValidType> {
    pub(super) size: usize,
    pub(super) ptr: *mut T,
    pub(super) ctx: Arc<ExecutionContext>,
}

impl<T: ValidType> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            util::free(self.ptr, self.size).unwrap();
        }
    }
}

impl<T: ValidType> Buffer<T> {
    pub fn ptr(&self) -> *const T {
        self.ptr.cast_const()
    }

    pub fn mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T: ValidType> DeviceBuffer<ExecutionContext, T> for Buffer<T> {
    type BufferError = DeviceError;

    /// Creates a new **zeroed** buffer with the given number of elements.
    fn new(ctx: Arc<ExecutionContext>, size: usize) -> Result<Self, DeviceError> {
        Ok(Self { size, ptr: unsafe { util::calloc(size) }?, ctx })
    }

    fn device(&self) -> Arc<ExecutionContext> {
        self.ctx.clone()
    }

    fn size(&self) -> usize {
        self.size
    }

    fn set_zero(&mut self) -> Result<(), DeviceError> {
        unsafe { util::set_zero(self.ptr, self.size) }
    }

    fn load_from_device(&mut self, buf: &Self, bytes: usize) -> Result<(), DeviceError> {
        assert!(bytes <= buf.size, "Overflow: {bytes} > {}!", buf.size);
        assert!(bytes <= self.size, "Overflow: {} > {}!", buf.size, self.size);
        unsafe { util::copy_on_device(self.ptr, buf.ptr, bytes) }
    }

    fn load_from_slice(&mut self, buf: &[T]) -> Result<(), DeviceError> {
        assert!(buf.len() <= self.size, "Overflow!");
        unsafe { util::copy_to_device(self.ptr, buf.as_ptr(), buf.len()) }
    }

    unsafe fn load_non_blocking_from_host(&mut self, buf: &[T]) -> Result<(), Self::BufferError> {
        assert!(buf.len() <= self.size, "Overflow!");
        unsafe { util::async_copy_to_device(self.ptr, buf.as_ptr(), buf.len(), self.ctx.copystream) }
    }

    fn write_into_slice(&self, buf: &mut [T], bytes: usize) -> Result<(), DeviceError> {
        assert!(bytes <= self.size, "Overflow!");
        unsafe { util::copy_from_device(buf.as_mut_ptr(), self.ptr, bytes) }
    }
}

pub mod util {
    use crate::{backend::bindings::cudaStream_t, DeviceError};

    use super::super::{bindings, util::catch};
    use std::ffi::c_void;

    pub unsafe fn malloc<T>(num: usize) -> Result<*mut T, DeviceError> {
        let size = num * std::mem::size_of::<T>();
        let mut grad = std::ptr::null_mut::<T>();
        let grad_ptr = (&mut grad) as *mut *mut T;

        assert!(!grad_ptr.is_null(), "null pointer");

        unsafe {
            catch(bindings::cudaMalloc(grad_ptr.cast(), size))?;
            catch(bindings::cudaDeviceSynchronize())?;
        }

        Ok(grad)
    }

    /// ### Safety
    /// Need to make sure not to double free.
    pub unsafe fn free<T>(ptr: *mut T, _: usize) -> Result<(), DeviceError> {
        catch(bindings::cudaFree(ptr.cast()))
    }

    /// ### Safety
    /// Type needs to be zeroable.
    pub unsafe fn calloc<T>(num: usize) -> Result<*mut T, DeviceError> {
        let size = num * std::mem::size_of::<T>();
        let grad = malloc(num)?;
        catch(bindings::cudaMemset(grad as *mut c_void, 0, size))?;
        catch(bindings::cudaDeviceSynchronize())?;

        Ok(grad)
    }

    /// ### Safety
    /// Type needs to be zeroable.
    pub unsafe fn set_zero<T>(ptr: *mut T, num: usize) -> Result<(), DeviceError> {
        catch(bindings::cudaMemset(ptr.cast(), 0, num * std::mem::size_of::<T>()))
    }

    /// # Safety
    /// Pointers need to be valid and `amt` need to be valid.
    pub unsafe fn copy_to_device<T>(dest: *mut T, src: *const T, amt: usize) -> Result<(), DeviceError> {
        catch(bindings::cudaMemcpy(dest.cast(), src.cast(), amt * std::mem::size_of::<T>(), bindings::H2D))?;
        catch(bindings::cudaDeviceSynchronize())
    }

    /// # Safety
    /// Pointers need to be valid and `amt` need to be valid.
    /// Data in `src` cannot be freed or modified before device is synchronised.
    pub unsafe fn async_copy_to_device<T>(
        dest: *mut T,
        src: *const T,
        amt: usize,
        stream: cudaStream_t,
    ) -> Result<(), DeviceError> {
        catch(bindings::cudaMemcpyAsync(dest.cast(), src.cast(), amt * std::mem::size_of::<T>(), bindings::H2D, stream))
    }

    /// # Safety
    /// Pointers need to be valid and `amt` need to be valid.
    pub unsafe fn copy_from_device<T>(dest: *mut T, src: *const T, amt: usize) -> Result<(), DeviceError> {
        catch(bindings::cudaMemcpy(dest.cast(), src.cast(), amt * std::mem::size_of::<T>(), bindings::D2H))?;
        catch(bindings::cudaDeviceSynchronize())
    }

    /// # Safety
    /// Pointers need to be valid and `amt` need to be valid.
    pub unsafe fn copy_on_device<T>(dest: *mut T, src: *const T, amt: usize) -> Result<(), DeviceError> {
        catch(bindings::cudaMemcpy(dest.cast(), src.cast(), amt * std::mem::size_of::<T>(), bindings::D2D))?;
        catch(bindings::cudaDeviceSynchronize())
    }
}
