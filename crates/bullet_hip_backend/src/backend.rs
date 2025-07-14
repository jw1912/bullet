pub mod bindings;
pub mod blas;
mod buffer;
pub mod ops;
pub mod util;

use std::sync::{Arc, Mutex};

use bindings::{cublasHandle_t, cudaStream_t};
pub use buffer::Buffer;

/// This contains the internal environment for the GPU to use
#[derive(Debug)]
pub struct ExecutionContext {
    cublas: cublasHandle_t,
    copystream: cudaStream_t,
    ones: Mutex<(usize, *mut f32)>,
}

impl ExecutionContext {
    pub fn with_ones<T, F: FnMut(&Buffer<f32>) -> T>(self: Arc<Self>, count: usize, mut f: F) -> T {
        let clone = self.clone();

        let mut ones = self.ones.try_lock().unwrap();

        if count > ones.0 {
            if !ones.1.is_null() {
                unsafe { buffer::util::free(ones.1, ones.0).unwrap() };
            }

            ones.0 = count;
            ones.1 = unsafe { buffer::util::malloc(count).unwrap() };
            unsafe {
                ops::set(ones.1, count, 1.0);
            }
        }

        let ones = Buffer { ptr: ones.1, size: count, ctx: clone };

        let res = f(&ones);

        std::mem::forget(ones);

        res
    }
}

unsafe impl Send for ExecutionContext {}
unsafe impl Sync for ExecutionContext {}

impl Drop for ExecutionContext {
    fn drop(&mut self) {
        unsafe {
            let status = bindings::cudaDeviceSynchronize();
            assert_eq!(status, bindings::SUCCESS);

            let status = bindings::cublasDestroy_v2(self.cublas);
            assert_eq!(status, bindings::CUBLAS_SUCCESS);

            let status = bindings::cudaStreamDestroy(self.copystream);
            assert_eq!(status, bindings::SUCCESS);

            let status = bindings::cudaDeviceSynchronize();
            assert_eq!(status, bindings::SUCCESS);
        }
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        let mut cublas: cublasHandle_t = std::ptr::null_mut();
        let mut copystream: cudaStream_t = std::ptr::null_mut();

        unsafe {
            let status = bindings::cublasCreate_v2((&mut cublas) as *mut cublasHandle_t);
            assert_eq!(status, bindings::CUBLAS_SUCCESS);

            let status = bindings::cudaStreamCreateWithFlags((&mut copystream) as *mut cudaStream_t, 1);
            assert_eq!(status, bindings::SUCCESS);

            let status = bindings::cudaDeviceSynchronize();
            assert_eq!(status, bindings::SUCCESS);
        }

        Self { cublas, copystream, ones: Mutex::new((0, std::ptr::null_mut())) }
    }
}
