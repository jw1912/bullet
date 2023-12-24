use std::sync::atomic::{AtomicUsize, Ordering, AtomicBool};

use crate::util;

static ALLOC_ID: AtomicUsize = AtomicUsize::new(0);
static TRACKING: AtomicBool = AtomicBool::new(false);

/// Managed memory buffer of single-precision floats on the GPU.
pub struct GpuBuffer {
    size: usize,
    ptr: *mut f32,
    id: usize,
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        self.report("Freed");
        unsafe {
            util::free(self.ptr.cast());
        }
    }
}

impl GpuBuffer {
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
        self.report("Accessed");
        self.ptr
    }

    pub fn load_from_cpu(&self, buf: &[f32]) {
        assert!(buf.len() == self.size, "Must be exactly the same size!");
        util::copy_to_gpu(self.ptr, buf.as_ptr(), self.size);
        self.report("Loaded from CPU");
    }

    pub fn write_to_cpu(&self, buf: &mut [f32]) {
        assert!(buf.len() == self.size, "Must be exactly the same size!");
        util::copy_from_gpu(buf.as_mut_ptr(), self.ptr, self.size);
        self.report("Wrote to CPU");
    }

    fn report(&self, msg: &str) {
        if TRACKING.load(Ordering::SeqCst) {
            println!("[CUDA#{}] {msg}" ,self.id);
        }
    }
}