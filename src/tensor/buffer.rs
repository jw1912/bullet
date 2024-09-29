use crate::backend::util;

/// Managed memory buffer of `T` on the device.
#[derive(Debug)]
pub struct Buffer<T: Copy> {
    size: usize,
    ptr: *mut T,
}

impl<T: Copy> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            util::free(self.ptr, self.size);
        }
    }
}

impl<T: Copy> Buffer<T> {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            ptr: util::calloc(size),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn set_zero(&self) {
        util::set_zero(self.ptr, self.size)
    }

    pub fn load_from_device(&self, buf: &Self) {
        assert!(buf.size <= self.size, "Overflow: {} > {}!", buf.size, self.size);
        unsafe {
            util::copy_on_device(self.ptr, buf.ptr, buf.size);
        }
        util::device_synchronise();
    }

    pub fn load_from_slice(&self, buf: &[T]) {
        assert!(buf.len() <= self.size, "Overflow!");
        unsafe {
            util::copy_to_device(self.ptr, buf.as_ptr(), buf.len());
        }
        util::device_synchronise();
    }

    pub fn write_into_slice(&self, buf: &mut [T]) {
        assert!(buf.len() <= self.size, "Overflow!");
        unsafe {
            util::copy_from_device(buf.as_mut_ptr(), self.ptr, buf.len());
        }
        util::device_synchronise();
    }
}
