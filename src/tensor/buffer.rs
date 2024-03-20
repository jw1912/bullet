use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::backend::util;

static ALLOC_ID: AtomicUsize = AtomicUsize::new(0);
static TRACKING: AtomicBool = AtomicBool::new(false);

/// Managed memory buffer of single-precision floats on the GPU.
pub struct DeviceBuffer {
    size: usize,
    ptr: *mut f32,
    id: usize,
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        self.report("Freed");
        unsafe {
            util::free(self.ptr, self.size);
        }
    }
}

impl DeviceBuffer {
    pub fn new(size: usize) -> Self {
        ALLOC_ID.fetch_add(1, Ordering::SeqCst);
        let id = ALLOC_ID.load(Ordering::SeqCst);

        let res = Self {
            size,
            ptr: util::calloc(size),
            id,
        };

        res.report("Allocated");

        res
    }

    pub fn set_tracking(tracking: bool) {
        TRACKING.store(tracking, Ordering::SeqCst);
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn ptr(&self) -> *mut f32 {
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

    pub fn load_from_host(&self, buf: &[f32]) {
        assert!(buf.len() <= self.size, "Overflow!");
        unsafe {
            util::copy_to_device(self.ptr, buf.as_ptr(), buf.len());
        }
        util::device_synchronise();
    }

    pub fn write_to_host(&self, buf: &mut [f32]) {
        assert!(buf.len() <= self.size, "Overflow!");
        unsafe {
            util::copy_from_device(buf.as_mut_ptr(), self.ptr, self.size);
        }
        util::device_synchronise();
    }

    fn report(&self, msg: &str) {
        if TRACKING.load(Ordering::SeqCst) {
            println!("[CUDA#{}] {msg}", self.id);
        }
    }
}
