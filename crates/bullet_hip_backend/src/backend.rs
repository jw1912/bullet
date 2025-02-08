pub mod bindings;
pub mod blas;
mod buffer;
pub mod ops;
pub mod util;

use bindings::cublasHandle_t;
pub use buffer::Buffer;

/// This contains the internal environment for the GPU to use
#[derive(Debug)]
pub struct ExecutionContext {
    cublas: cublasHandle_t,
}

unsafe impl Send for ExecutionContext {}
unsafe impl Sync for ExecutionContext {}

impl Drop for ExecutionContext {
    fn drop(&mut self) {
        unsafe {
            let status = bindings::cublasDestroy_v2(self.cublas);
            assert_eq!(status, bindings::CUBLAS_SUCCESS);
        }
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        let mut cublas: cublasHandle_t = std::ptr::null_mut();

        unsafe {
            let status = bindings::cublasCreate_v2((&mut cublas) as *mut cublasHandle_t);
            assert_eq!(status, bindings::CUBLAS_SUCCESS);
        }

        Self { cublas }
    }
}
