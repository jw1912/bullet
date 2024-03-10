use crate::backend::util;
use super::Shape;

/// Single Rank-2 Tensor on the GPU.
/// This data type does not own the memory it points to,
/// it must be manually allocated and freed, or used to
/// borrow data only.
pub struct Tensor {
    shape: Shape,
    ptr: *mut f32,
}

impl Tensor {
    /// # Safety
    /// This creates an uninitialised instance, it is up to the
    /// user to perform an operation which initialises it.
    pub unsafe fn uninit(shape: Shape) -> Self {
        Self {
            shape,
            ptr: std::ptr::null_mut(),
        }
    }

    /// # Safety
    /// You can set this to point to anywhere.
    pub unsafe fn set_ptr(&mut self, ptr: *mut f32) {
        self.ptr = ptr;
    }

    pub fn calloc(&mut self) {
        self.ptr = util::calloc(self.num_elements());
    }

    /// # Safety
    /// Don't double free.
    pub unsafe fn free(&mut self) {
        util::free(self.ptr.cast(), self.num_elements());
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn ptr(&self) -> *mut f32 {
        self.ptr
    }

    pub fn reshape(&mut self, cols: usize, rows: usize) {
        self.shape.reshape(cols, rows);
    }

    pub fn num_elements(&self) -> usize {
        self.shape.size()
    }

    pub fn load_from_host(&self, buf: &[f32]) {
        assert!(
            !self.ptr.is_null(),
            "Attempting to dereference null pointer!"
        );

        assert!(
            buf.len() == self.num_elements(),
            "Must be exactly the same size!"
        );

        unsafe {
            util::copy_to_device(self.ptr, buf.as_ptr(), buf.len());
        }
    }

    pub fn write_to_host(&self, buf: &mut [f32]) {
        assert!(
            !self.ptr.is_null(),
            "Attempting to dereference null pointer!"
        );

        assert!(
            buf.len() == self.num_elements(),
            "Must be exactly the same size!"
        );

        unsafe {
            util::copy_from_device(buf.as_mut_ptr(), self.ptr, buf.len());
        }
    }
}
