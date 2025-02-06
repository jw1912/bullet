use std::sync::Arc;

use bullet_core::device::DeviceBuffer;

use super::ExecutionContext;

/// # Safety
/// Don't impl this for anything else
pub unsafe trait ValidType {}
unsafe impl ValidType for f32 {}
unsafe impl ValidType for i32 {}

/// Managed memory buffer of `T` on the device.
#[derive(Debug)]
pub struct Buffer<T: ValidType> {
    size: usize,
    ptr: *mut T,
    ctx: Arc<ExecutionContext>,
}

impl<T: ValidType> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            util::free(self.ptr, self.size);
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
    /// Creates a new **zeroed** buffer with the given number of elements.
    fn new(ctx: Arc<ExecutionContext>, size: usize) -> Self {
        Self { size, ptr: unsafe { util::calloc(size) }, ctx }
    }

    fn device(&self) -> Arc<ExecutionContext> {
        self.ctx.clone()
    }

    fn size(&self) -> usize {
        self.size
    }

    fn set_zero(&mut self) {
        unsafe {
            util::set_zero(self.ptr, self.size);
        }
    }

    fn load_from_device(&mut self, buf: &Self, bytes: usize) {
        assert!(bytes <= buf.size);
        assert!(bytes <= self.size, "Overflow: {} > {}!", buf.size, self.size);
        unsafe {
            util::copy_on_device(self.ptr, buf.ptr, bytes);
        }
    }

    fn load_from_slice(&mut self, buf: &[T]) {
        assert!(buf.len() <= self.size, "Overflow!");
        unsafe {
            util::copy_to_device(self.ptr, buf.as_ptr(), buf.len());
        }
    }

    fn write_into_slice(&self, buf: &mut [T], bytes: usize) {
        assert!(bytes <= self.size, "Overflow!");
        unsafe {
            util::copy_from_device(buf.as_mut_ptr(), self.ptr, bytes);
        }
    }
}

mod util {
    use super::super::{bindings, util::catch};
    use std::ffi::c_void;

    fn malloc<T>(num: usize) -> *mut T {
        let size = num * std::mem::size_of::<T>();
        let mut grad = std::ptr::null_mut::<T>();
        let grad_ptr = (&mut grad) as *mut *mut T;

        assert!(!grad_ptr.is_null(), "null pointer");

        unsafe {
            catch(bindings::cudaMalloc(grad_ptr.cast(), size), "Malloc");
            catch(bindings::cudaDeviceSynchronize(), "DeviceSynchronize");
        }

        grad
    }

    /// ### Safety
    /// Need to make sure not to double free.
    pub unsafe fn free<T>(ptr: *mut T, _: usize) {
        catch(bindings::cudaFree(ptr.cast()), "Free");
    }

    /// ### Safety
    /// Type needs to be zeroable.
    pub unsafe fn calloc<T>(num: usize) -> *mut T {
        let size = num * std::mem::size_of::<T>();
        let grad = malloc(num);
        catch(bindings::cudaMemset(grad as *mut c_void, 0, size), "Memset");
        catch(bindings::cudaDeviceSynchronize(), "DeviceSynchronize");

        grad
    }

    /// ### Safety
    /// Type needs to be zeroable.
    pub unsafe fn set_zero<T>(ptr: *mut T, num: usize) {
        catch(bindings::cudaMemset(ptr.cast(), 0, num * std::mem::size_of::<T>()), "memset");
        catch(bindings::cudaDeviceSynchronize(), "DeviceSynchronize");
    }

    /// # Safety
    /// Pointers need to be valid and `amt` need to be valid.
    pub unsafe fn copy_to_device<T>(dest: *mut T, src: *const T, amt: usize) {
        catch(bindings::cudaMemcpy(dest.cast(), src.cast(), amt * std::mem::size_of::<T>(), bindings::H2D), "Memcpy");
        catch(bindings::cudaDeviceSynchronize(), "DeviceSynchronize");
    }

    /// # Safety
    /// Pointers need to be valid and `amt` need to be valid.
    pub unsafe fn copy_from_device<T>(dest: *mut T, src: *const T, amt: usize) {
        catch(bindings::cudaMemcpy(dest.cast(), src.cast(), amt * std::mem::size_of::<T>(), bindings::D2H), "Memcpy");
        catch(bindings::cudaDeviceSynchronize(), "DeviceSynchronize");
    }

    /// # Safety
    /// Pointers need to be valid and `amt` need to be valid.
    pub unsafe fn copy_on_device<T>(dest: *mut T, src: *const T, amt: usize) {
        catch(bindings::cudaMemcpy(dest.cast(), src.cast(), amt * std::mem::size_of::<T>(), bindings::D2D), "Memcpy");
        catch(bindings::cudaDeviceSynchronize(), "DeviceSynchronize");
    }
}
